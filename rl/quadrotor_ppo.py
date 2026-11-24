"""PPO for the smooth quadrotor comparison, with bounded rotor actions.

PPO likelihood ratios are evaluated in the pre-tanh Gaussian coordinates. The
Jacobian of the common bijection cancels in the ratio; actions are never sampled
from a Gaussian and then clipped for likelihood evaluation.
"""

from dataclasses import dataclass
from typing import NamedTuple

import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from envs.quadrotor_smooth import MASS, GRAVITY


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 262144
    num_envs: int = 32
    rollout_steps: int = 128
    epochs: int = 4
    minibatches: int = 4
    layers: int = 3
    width: int = 64
    learning_rate: float = 3e-4
    gamma: float = .995
    gae_lambda: float = .95
    clip: float = .2
    value_coef: float = .5
    entropy_coef: float = 0.
    max_grad_norm: float = .5

    def validate(self):
        positive = [self.total_steps, self.num_envs, self.rollout_steps,
                    self.epochs, self.minibatches, self.layers, self.width]
        if any(x <= 0 for x in positive):
            raise ValueError("Step, network and batch sizes must be positive")
        if self.batch_size % self.minibatches or self.total_steps % self.batch_size:
            raise ValueError("Minibatches must divide a rollout; total steps must be a multiple of the rollout batch")
        if not 0 < self.learning_rate or not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid PPO learning rate, discount or GAE lambda")

    @property
    def batch_size(self):
        return self.num_envs * self.rollout_steps

    @property
    def updates(self):
        return self.total_steps // self.batch_size


class QuadActorCritic(nn.Module):
    layers: int
    width: int
    initial_action_mean: tuple | None = None

    @nn.compact
    def __call__(self, obs):
        actor, critic = obs, obs
        for i in range(self.layers):
            actor = nn.leaky_relu(nn.Dense(self.width, name=f"actor_{i}",
                kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(actor))
            critic = nn.leaky_relu(nn.Dense(self.width, name=f"critic_{i}",
                kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(critic))
        hover = MASS * GRAVITY  # allocation: four .25 coefficients, equal commands
        bias = (float(np.arctanh(2 * hover - 1)) if self.initial_action_mean is None
                else np.arctanh(2 * np.asarray(self.initial_action_mean) - 1))
        mean = nn.Dense(4, name="actor_out", kernel_init=nn.initializers.orthogonal(.01),
                        bias_init=nn.initializers.constant(bias))(actor)
        logstd = self.param("logstd", nn.initializers.constant(-1.4), (4,))
        logstd = jnp.clip(logstd, -5., 1.)
        value = nn.Dense(1, name="value", kernel_init=nn.initializers.orthogonal(1.))(critic)
        return mean, logstd, value[..., 0]


def command(latent):
    return (jnp.tanh(latent) + 1) / 2


def log_probability(latent, mean, logstd):
    return jnp.sum(-.5 * ((latent - mean) * jnp.exp(-logstd)) ** 2
                   - logstd - .5 * jnp.log(2 * jnp.pi), axis=-1)


def advantages_and_returns(reward, value, next_value, terminated, truncated, gamma, lam):
    """Bootstrap time limits, but never continue GAE across an episode reset."""
    delta = reward + gamma * next_value * (1 - terminated) - value
    continuation = 1 - (terminated | truncated)
    def body(gae, values):
        d, cont = values
        gae = d + gamma * lam * cont * gae
        return gae, gae
    _, advantages = jax.lax.scan(body, jnp.zeros_like(value[0]),
                                 (delta, continuation), reverse=True)
    return advantages, advantages + value


class Runner(NamedTuple):
    train_state: TrainState
    env_state: object
    obs: jax.Array
    action_rng: jax.Array
    env_rng: jax.Array
    episode_return: jax.Array


def make_learner(env, config, seed, *, actor_auxiliary_loss=None):
    config.validate()
    reward_scale = getattr(env, "learning_reward_scale", 1.)
    model = QuadActorCritic(config.layers, config.width,
                            initial_action_mean=getattr(env, "initial_action_mean", None))
    # These key streams are paired across variants regardless of observation size.
    init_key, action_key = jax.random.split(jax.random.PRNGKey(seed))
    env_key = jax.random.PRNGKey(seed + 10000)
    params = model.init(init_key, jnp.zeros((env.obs_size,)))
    schedule = optax.linear_schedule(config.learning_rate, 0., config.updates * config.epochs * config.minibatches)
    tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(schedule, eps=1e-5))
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    env_key, reset_key = jax.random.split(env_key)
    states, obs = jax.vmap(env.reset)(jax.random.split(reset_key, config.num_envs))
    runner = Runner(train_state, states, obs, action_key, env_key, jnp.zeros(config.num_envs))

    def update(runner):
        def collect(carry, _):
            states, obs, action_rng, env_rng, episode_return = carry
            action_rng, draw_key = jax.random.split(action_rng)
            mean, logstd, value = model.apply(runner.train_state.params, obs)
            latent = mean + jnp.exp(logstd) * jax.random.normal(draw_key, mean.shape)
            logp = log_probability(latent, mean, logstd)
            next_states, terminal_obs, reward, failed, truncated, info = jax.vmap(env.transition)(states, command(latent))
            _, _, next_value = model.apply(runner.train_state.params, terminal_obs)
            done = failed | truncated
            # Reset every lane with exogenous keys, then select only ended episodes.
            env_rng, reset_key = jax.random.split(env_rng)
            reset_states, reset_obs = jax.vmap(env.reset)(jax.random.split(reset_key, config.num_envs))
            next_states = jax.tree_util.tree_map(
                lambda new, old: jnp.where(done.reshape(done.shape + (1,) * (old.ndim - 1)), new, old),
                reset_states, next_states)
            next_obs = jnp.where(done[:, None], reset_obs, terminal_obs)
            ep_return = episode_return + reward
            transition = {"obs": obs, "latent": latent, "logp": logp,
                          "value": value, "next_value": next_value, "reward": reward,
                          "failed": failed, "truncated": truncated,
                          "ended_return": jnp.where(done, ep_return, 0.), "done": done,
                          "position_sq": info["position_sq"], "velocity_sq": info["velocity_sq"],
                          "attitude_sq": info["attitude_sq"],
                          **{name: info[name] for name in ["position_cost", "velocity_cost", "attitude_cost",
                                                          "omega_cost", "rotor_command_cost", "failure_cost"]}}
            return (next_states, next_obs, action_rng, env_rng, jnp.where(done, 0., ep_return)), transition

        initial = (runner.env_state, runner.obs, runner.action_rng, runner.env_rng, runner.episode_return)
        (states, obs, action_rng, env_rng, ep_return), batch = jax.lax.scan(
            collect, initial, None, length=config.rollout_steps)
        # Value predictions/targets use scaled units to condition the critic.
        # Raw rewards, episode returns and reward-component logs remain unscaled.
        learning_reward = batch["reward"] if reward_scale == 1. else batch["reward"] / reward_scale
        adv, target = advantages_and_returns(learning_reward, batch["value"], batch["next_value"],
                                             batch["failed"], batch["truncated"], config.gamma, config.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch["adv"], batch["target"] = adv, target
        flat = jax.tree_util.tree_map(lambda x: x.reshape((config.batch_size,) + x.shape[2:]), batch)

        def loss(params, minibatch):
            mean, logstd, value = model.apply(params, minibatch["obs"])
            logp = log_probability(minibatch["latent"], mean, logstd)
            ratio = jnp.exp(logp - minibatch["logp"])
            policy_loss = -jnp.minimum(ratio * minibatch["adv"],
                jnp.clip(ratio, 1 - config.clip, 1 + config.clip) * minibatch["adv"]).mean()
            clipped_value = minibatch["value"] + jnp.clip(value - minibatch["value"], -config.clip, config.clip)
            value_loss = .5 * jnp.maximum((value - minibatch["target"]) ** 2,
                                           (clipped_value - minibatch["target"]) ** 2).mean()
            # Latent Gaussian entropy; entropy_coef=0 in the comparison presets.
            latent_entropy = jnp.sum(logstd + .5 * jnp.log(2 * jnp.pi * jnp.e))
            total = policy_loss + config.value_coef * value_loss - config.entropy_coef * latent_entropy
            if actor_auxiliary_loss is not None:
                total += actor_auxiliary_loss(model, params)
            kl = jnp.mean((ratio - 1) - (logp - minibatch["logp"]))
            return total, jnp.stack([policy_loss, value_loss, kl])

        def epoch(carry, _):
            train_state, key = carry
            key, shuffle_key = jax.random.split(key)
            permutation = jax.random.permutation(shuffle_key, config.batch_size)
            minibatches = jax.tree_util.tree_map(lambda x: x[permutation].reshape(
                (config.minibatches, config.batch_size // config.minibatches) + x.shape[1:]), flat)
            def optimize(train_state, minibatch):
                (_, stats), gradients = jax.value_and_grad(loss, has_aux=True)(train_state.params, minibatch)
                return train_state.apply_gradients(grads=gradients), stats
            train_state, stats = jax.lax.scan(optimize, train_state, minibatches)
            return (train_state, key), stats

        (train_state, action_rng), stats = jax.lax.scan(epoch, (runner.train_state, action_rng), None, length=config.epochs)
        ended = batch["done"].sum()
        metrics = {"step_reward": batch["reward"].mean(),
                   "learning_step_reward": learning_reward.mean(),
                   "rollout_position_rmse": jnp.sqrt(batch["position_sq"].mean()),
                   "rollout_velocity_rmse": jnp.sqrt(batch["velocity_sq"].mean()),
                   "rollout_attitude_rmse": jnp.sqrt(batch["attitude_sq"].mean()),
                   "completed_episodes": ended,
                   "episode_return_sum": batch["ended_return"].sum(),
                   "failures": batch["failed"].sum(), "policy_loss": stats[..., 0].mean(),
                   "value_loss": stats[..., 1].mean(), "approx_kl": stats[..., 2].mean()}
        metrics.update({"mean_" + name: batch[name].mean() for name in
                        ["position_cost", "velocity_cost", "attitude_cost", "omega_cost",
                         "rotor_command_cost", "failure_cost"]})
        return Runner(train_state, states, obs, action_rng, env_rng, ep_return), metrics

    return model, runner, jax.jit(update)


def make_evaluator(env, model, seed=900000, record_state=False, *, initial_states=None):
    """Deterministic policy, fixed paired starts; stop at first failure.

    Optional state recording enables later path/attitude/control plots without
    rerunning the policy. It does not change the rollout or summary metrics.
    """
    count = env.bank.pos.shape[0]
    if initial_states is None:
        keys = jax.random.split(jax.random.PRNGKey(seed), count)
        states, _ = jax.vmap(env.reset)(keys, jnp.arange(count))
    else:
        states = initial_states

    def evaluate(params):
        def step(carry, step_index):
            states, active, totals = carry
            mean, _, _ = model.apply(params, jax.vmap(env.observe)(states))
            next_states, _, reward, failed, truncated, info = jax.vmap(env.transition)(states, command(mean))
            values = jnp.stack([info["position_sq"], info["velocity_sq"], info["attitude_sq"],
                                info["omega_sq"], info["saturation"], info["command_rate_sq"], reward,
                                jnp.ones_like(reward)], axis=-1)
            totals = totals + jnp.where(active[:, None], values, 0.)
            # Freeze ended trajectories. Prevent reset jumps from entering tracking metrics.
            next_states = jax.tree_util.tree_map(lambda new, old: jnp.where(
                active.reshape(active.shape + (1,) * (old.ndim - 1)), new, old), next_states, states)
            next_active = active & ~(failed | truncated)
            frame = {"position": next_states.pos, "reference": env.bank.pos[:, step_index + 1],
                     "active": active, "failed": active & failed}
            if record_state:
                frame.update(velocity=next_states.vel, rotation=next_states.rot,
                             omega=next_states.omega, rotor_command=next_states.previous_action)
                if hasattr(next_states, "thrust"):
                    frame.update(thrust=next_states.thrust, thrust_rate=next_states.thrust_rate)
            return (next_states, next_active, totals), frame
        (_, _, totals), frames = jax.lax.scan(step, (states, jnp.ones(count, dtype=bool),
                                                     jnp.zeros((count, 8))), jnp.arange(env.horizon))
        failures = jnp.any(frames["failed"], axis=0)
        samples = jnp.maximum(totals[:, 7], 1.)
        per_trajectory = jnp.concatenate([
            jnp.sqrt(totals[:, :4] / samples[:, None]),
            (totals[:, 4] / samples)[:, None],
            jnp.sqrt(totals[:, 5] / samples)[:, None],
            totals[:, 6:7], failures[:, None].astype(jnp.float32),
            (samples * env.dt)[:, None]], axis=1)
        if record_state:
            frames.update(initial_position=states.pos, initial_velocity=states.vel,
                          initial_rotation=states.rot, initial_omega=states.omega)
            if hasattr(states, "thrust"):
                frames.update(initial_thrust=states.thrust, initial_thrust_rate=states.thrust_rate)
        return per_trajectory, frames
    return jax.jit(evaluate)


EVALUATION_COLUMNS = ["position_rmse_m", "velocity_rmse_mps", "attitude_rmse_rad",
                      "omega_rmse_radps", "saturation_fraction", "command_rate_rms_per_s",
                      "episode_return", "failed", "survival_seconds"]
