"""Relative Entropy Pathwise Policy Optimization for the quadrotor simulator."""

from dataclasses import dataclass
from typing import NamedTuple

import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr
import numpy as np
import optax

from rl.quadrotor_ppo import command, log_probability


ACTION_SIZE = 4
COSTS = ("position_cost", "velocity_cost", "attitude_cost", "omega_cost",
         "rotor_command_cost", "failure_cost")


@dataclass(frozen=True)
class ReppoConfig:
    total_steps: int
    num_envs: int = 32
    rollout_steps: int = 128
    epochs: int = 4
    minibatches: int = 4
    layers: int = 6
    width: int = 256
    learning_rate: float = 3e-4
    gamma: float = .995
    td_lambda: float = .95
    max_grad_norm: float = .5
    target_entropy: float = -.5 * ACTION_SIZE
    target_kl: float = .1
    initial_alpha: float = .01
    initial_beta: float = .01
    q_min: float = -40.
    q_max: float = 10.
    q_bins: int = 151
    auxiliary_weight: float = 1.

    @property
    def batch_size(self):
        return self.num_envs * self.rollout_steps

    @property
    def updates(self):
        return self.total_steps // self.batch_size

    def validate(self):
        if min(self.total_steps, self.num_envs, self.rollout_steps, self.epochs,
               self.minibatches, self.layers, self.width) < 1:
            raise ValueError("Step, batch and network sizes must be positive")
        if self.batch_size % self.minibatches or self.total_steps % self.batch_size:
            raise ValueError("Minibatches must divide a rollout and total steps must be a multiple of it")
        if not 0 < self.gamma <= 1 or not 0 <= self.td_lambda <= 1:
            raise ValueError("Invalid discount or TD lambda")
        positive = (self.learning_rate, self.max_grad_norm, self.target_kl, self.initial_alpha,
                    self.initial_beta)
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError("Learning rate, gradient norm, KL target and starting multipliers must be positive")
        if not np.isfinite(self.target_entropy) or self.target_entropy >= ACTION_SIZE * np.log(2):
            raise ValueError("Entropy target must be finite and below the log volume of the tanh action space")
        if (not np.isfinite((self.q_min, self.q_max, self.auxiliary_weight)).all()
                or self.q_min >= self.q_max or self.q_bins < 2 or self.auxiliary_weight < 0):
            raise ValueError("Invalid critic support or auxiliary loss weight")


class QuadActor(nn.Module):
    """Same Gaussian actor and named initializers as the PPO quadrotor actor."""

    layers: int
    width: int
    initial_action_mean: tuple

    @nn.compact
    def __call__(self, obs):
        hidden = obs
        for index in range(self.layers):
            hidden = nn.leaky_relu(nn.Dense(
                self.width, name=f"actor_{index}",
                kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(hidden))
        bias = np.arctanh(2 * np.asarray(self.initial_action_mean) - 1)
        mean = nn.Dense(ACTION_SIZE, name="actor_out", kernel_init=nn.initializers.orthogonal(.01),
                        bias_init=nn.initializers.constant(bias))(hidden)
        logstd = self.param("logstd", nn.initializers.constant(-1.4), (ACTION_SIZE,))
        return mean, jnp.clip(logstd, -5., 1.)


class NormalizedDense(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.width, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        return nn.silu(nn.LayerNorm()(x))


class QuadQCritic(nn.Module):
    layers: int
    width: int
    bins: int

    @nn.compact
    def __call__(self, obs, action, *, predict=False):
        encoder_depth = (self.layers + 1) // 2
        embedding = jnp.concatenate((obs, 2. * action - 1.), axis=-1)
        for index in range(encoder_depth):
            embedding = NormalizedDense(self.width, name=f"encoder_{index}")(embedding)
        hidden = embedding
        for index in range(self.layers - encoder_depth):
            hidden = NormalizedDense(self.width, name=f"q_hidden_{index}")(hidden)
        logits = nn.Dense(self.bins, kernel_init=nn.initializers.orthogonal(.01), name="q_logits")(hidden)
        prediction = None
        if predict:
            hidden = NormalizedDense(self.width, name="predict_hidden")(embedding)
            prediction = nn.Dense(self.width, name="predict_embedding")(hidden)
        return logits, embedding, prediction


def sample_latent(key, mean, logstd):
    return mean + jnp.exp(logstd) * jax.random.normal(key, mean.shape, mean.dtype)


def squashed_log_probability(latent, mean, logstd):
    """Log density on [-1, 1], evaluated before the affine map to simulator commands."""
    log_det = jnp.sum(2. * (jnp.log(2.) - latent - jax.nn.softplus(-2. * latent)), axis=-1)
    return log_probability(latent, mean, logstd) - log_det


def forward_kl(old_mean, old_logstd, mean, logstd):
    """KL(rollout policy || current policy), invariant to their common action map."""
    log_ratio = 2. * (old_logstd - logstd)
    diff = (old_mean - mean) * jnp.exp(-logstd)
    return .5 * jnp.sum(jnp.expm1(log_ratio) - log_ratio + jnp.square(diff), axis=-1)


def soft_td_lambda(reward, next_q, next_logp, failed, truncated, *, gamma, lam, alpha):
    """On-policy Q targets with next-action entropy and pre-reset bootstraps."""
    def backwards(next_return, transition):
        r, q, logp, failure, timeout = transition
        continuation = jnp.where(timeout, q, (1. - lam) * q + lam * next_return)
        target = r + gamma * jnp.where(failure, 0., continuation - alpha * logp)
        return target, target

    _, targets = jax.lax.scan(
        backwards, next_q[-1], (reward, next_q, next_logp, failed, truncated), reverse=True)
    return targets


def hl_gauss_targets(targets, centers):
    """Assign smoothed target mass to the critic's fixed return bins."""
    width = centers[1] - centers[0]
    edges = jnp.concatenate((centers[:1] - .5 * width, centers + .5 * width))
    clipped = jnp.clip(targets, centers[0], centers[-1])
    cdf = ndtr((edges - clipped[..., None]) / (.75 * width))
    return jnp.diff(cdf, axis=-1) / (cdf[..., -1:] - cdf[..., :1])


def expected_q(logits, centers):
    return jax.nn.softmax(logits, axis=-1) @ centers


def constrained_actor_loss(q, logp, kl, log_alpha, log_beta, config):
    """Paper Eq. 13 and dual updates, with gradients separated by parameter."""
    stop = jax.lax.stop_gradient
    alpha, beta = jnp.exp(log_alpha), jnp.exp(log_beta)
    loss = jnp.where(kl < config.target_kl, stop(alpha) * logp - q, stop(beta) * kl).mean()
    entropy_loss = alpha * stop(-logp.mean() - config.target_entropy)
    kl_loss = beta * stop(config.target_kl - kl.mean())
    return loss + entropy_loss + kl_loss, loss


class Runner(NamedTuple):
    actor: TrainState
    critic: TrainState
    env_state: object
    obs: jax.Array
    action_rng: jax.Array
    env_rng: jax.Array
    episode_return: jax.Array


def make_learner(env, config):
    config.validate()
    predict = config.auxiliary_weight > 0.
    actor = QuadActor(config.layers, config.width, tuple(env.initial_action_mean))
    critic = QuadQCritic(config.layers, config.width, config.q_bins)
    centers = jnp.linspace(config.q_min, config.q_max, config.q_bins)
    env_reset, env_transition = jax.vmap(env.reset), jax.vmap(env.transition)
    schedule = optax.linear_schedule(config.learning_rate, 0., config.updates * config.epochs * config.minibatches)
    optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(schedule, eps=1e-5))

    def initialize(seed):
        init_key, action_key = jax.random.split(jax.random.PRNGKey(seed))
        env_key, reset_key = jax.random.split(jax.random.PRNGKey(seed + 10000))
        sample_obs = jnp.zeros(env.obs_size)
        actor_params = {
            "policy": actor.init(init_key, sample_obs)["params"],
            "log_alpha": jnp.log(jnp.asarray(config.initial_alpha, dtype=sample_obs.dtype)),
            "log_beta": jnp.log(jnp.asarray(config.initial_beta, dtype=sample_obs.dtype)),
        }
        critic_params = critic.init(jax.random.fold_in(init_key, 1), sample_obs,
                                    jnp.full(ACTION_SIZE, .5), predict=predict)
        states, obs = env_reset(jax.random.split(reset_key, config.num_envs))
        return Runner(TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optimizer),
                      TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optimizer),
                      states, obs, action_key, env_key, jnp.zeros(config.num_envs))

    def update(runner):
        behavior = {"params": runner.actor.params["policy"]}
        alpha = jnp.exp(runner.actor.params["log_alpha"])

        def collect(carry, _):
            states, obs, action_rng, env_rng, episode_return = carry
            action_rng, draw_key = jax.random.split(action_rng)
            mean, logstd = actor.apply(behavior, obs)
            latent = sample_latent(draw_key, mean, logstd)
            action = command(latent)
            next_states, final_obs, reward, failed, truncated, info = env_transition(states, action)
            done = failed | truncated
            env_rng, reset_key = jax.random.split(env_rng)
            reset_states, reset_obs = env_reset(jax.random.split(reset_key, config.num_envs))
            next_states = jax.tree_util.tree_map(
                lambda new, old: jnp.where(done.reshape(done.shape + (1,) * (old.ndim - 1)), new, old),
                reset_states, next_states)
            next_obs = jnp.where(done[:, None], reset_obs, final_obs)
            episode_return = episode_return + reward
            transition = dict(obs=obs, mean=mean, logstd=jnp.broadcast_to(logstd, mean.shape),
                              latent=latent, action=action, final_obs=final_obs, reward=reward,
                              failed=failed, truncated=truncated, done=done,
                              ended_return=jnp.where(done, episode_return, 0.),
                              **{name: info[name] for name in ("position_sq", "velocity_sq", "attitude_sq", *COSTS)})
            return (next_states, next_obs, action_rng, env_rng, jnp.where(done, 0., episode_return)), transition

        initial = (runner.env_state, runner.obs, runner.action_rng, runner.env_rng, runner.episode_return)
        (states, obs, action_rng, env_rng, episode_return), batch = jax.lax.scan(
            collect, initial, None, length=config.rollout_steps)

        # Pair the next state and action from the rollout within each episode.
        # A fresh behavior-policy action supplies the bootstrap at rollout ends and time limits.
        safe_next_obs = jnp.where(batch["failed"][..., None], batch["obs"], batch["final_obs"])
        next_mean, next_logstd = actor.apply(behavior, safe_next_obs)
        bootstrap_latent = sample_latent(jax.random.fold_in(action_rng, 0), next_mean, next_logstd)
        shifted_latent = jnp.concatenate((batch["latent"][1:], batch["latent"][-1:]), axis=0)
        boundary = batch["done"].at[-1].set(True)
        next_latent = jnp.where(boundary[..., None], bootstrap_latent, shifted_latent)
        next_logp = squashed_log_probability(next_latent, next_mean, next_logstd)
        next_logits, next_embedding, _ = critic.apply(runner.critic.params, safe_next_obs, command(next_latent))
        reward = batch["reward"] / env.learning_reward_scale
        targets = soft_td_lambda(reward, expected_q(next_logits, centers), next_logp,
                                 batch["failed"], batch["truncated"],
                                 gamma=config.gamma, lam=config.td_lambda, alpha=alpha)
        training = {name: batch[name] for name in ("obs", "action", "mean", "logstd")}
        training.update(target=targets, histogram=hl_gauss_targets(targets, centers))
        if predict:
            training.update(next_embedding=next_embedding, continues=~batch["done"])
        flat = jax.tree_util.tree_map(lambda x: x.reshape((config.batch_size,) + x.shape[2:]), training)

        def optimize(carry, data):
            actor_state, critic_state = carry
            indices, sample_key = data
            minibatch = jax.tree_util.tree_map(lambda x: x[indices], flat)
            observations = minibatch["obs"]

            def critic_loss(params):
                logits, _, prediction = critic.apply(params, observations, minibatch["action"], predict=predict)
                cross_entropy = optax.softmax_cross_entropy(logits, minibatch["histogram"]).mean()
                auxiliary_loss = jnp.zeros_like(cross_entropy)
                if predict:
                    error = jnp.square(prediction - minibatch["next_embedding"]).mean(axis=-1)
                    valid = minibatch["continues"]
                    auxiliary_loss = jnp.where(valid, error, 0.).sum() / jnp.maximum(valid.sum(), 1)
                mse = jnp.square(expected_q(logits, centers) - minibatch["target"]).mean()
                return cross_entropy + config.auxiliary_weight * auxiliary_loss, {
                    "critic_cross_entropy": cross_entropy, "critic_auxiliary_loss": auxiliary_loss, "q_mse": mse}

            (_, critic_stats), grads = jax.value_and_grad(critic_loss, has_aux=True)(critic_state.params)
            critic_state = critic_state.apply_gradients(grads=grads)

            def actor_loss(params):
                mean, logstd = actor.apply({"params": params["policy"]}, observations)
                latent = sample_latent(sample_key, mean, logstd)
                logp = squashed_log_probability(latent, mean, logstd)
                logits, _, _ = critic.apply(critic_state.params, observations, command(latent))
                q = expected_q(logits, centers)
                kl = forward_kl(minibatch["mean"], minibatch["logstd"], mean, logstd)
                total, loss = constrained_actor_loss(q, logp, kl, params["log_alpha"], params["log_beta"], config)
                return total, {"actor_loss": loss, "entropy": -logp.mean(), "forward_kl": kl.mean(),
                               "kl_clip_fraction": (kl >= config.target_kl).mean()}

            (_, actor_stats), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
            actor_state = actor_state.apply_gradients(grads=grads)
            return (actor_state, critic_state), {**critic_stats, **actor_stats}

        def epoch(carry, _):
            actor_state, critic_state, key = carry
            key, shuffle_key = jax.random.split(key)
            indices = jax.random.permutation(shuffle_key, config.batch_size).reshape(config.minibatches, -1)
            keys = jax.random.split(jax.random.fold_in(shuffle_key, 1), config.minibatches)
            (actor_state, critic_state), stats = jax.lax.scan(optimize, (actor_state, critic_state), (indices, keys))
            return (actor_state, critic_state, key), stats

        (actor_state, critic_state, action_rng), stats = jax.lax.scan(
            epoch, (runner.actor, runner.critic, action_rng), None, length=config.epochs)
        metrics = dict(
            step_reward=batch["reward"].mean(), learning_step_reward=reward.mean(),
            rollout_position_rmse=jnp.sqrt(batch["position_sq"].mean()),
            rollout_velocity_rmse=jnp.sqrt(batch["velocity_sq"].mean()),
            rollout_attitude_rmse=jnp.sqrt(batch["attitude_sq"].mean()),
            completed_episodes=batch["done"].sum(), episode_return_sum=batch["ended_return"].sum(),
            failures=batch["failed"].sum(), alpha=jnp.exp(actor_state.params["log_alpha"]),
            beta=jnp.exp(actor_state.params["log_beta"]),
            q_target_min=targets.min(), q_target_max=targets.max(), q_target_mean=targets.mean(),
            q_target_below_range=(targets < config.q_min).mean(),
            q_target_above_range=(targets > config.q_max).mean(),
            **{f"mean_{name}": batch[name].mean() for name in COSTS},
            **jax.tree_util.tree_map(jnp.mean, stats),
        )
        return Runner(actor_state, critic_state, states, obs, action_rng, env_rng, episode_return), metrics

    return actor, initialize, update
