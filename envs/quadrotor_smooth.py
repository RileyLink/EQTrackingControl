"""Smooth, flatness-consistent quadrotor task for the policy comparison.

Independent of the legacy Gym/Orbax stack. Velocities are in world coordinates;
angular velocities are in each vehicle's body frame. Actions are the original
four normalized rotor commands, not command rates or residual controls.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


DT = 0.05
MASS = 0.067
GRAVITY = 9.81
INERTIA = jnp.diag(jnp.array([0.1, 0.1, 0.2]))
ALLOCATION = jnp.array([
    [0.25, 0.25, 0.25, 0.25],
    [0.0, -0.8, 0.0, 0.8],
    [0.8, 0.0, -0.8, 0.0],
    [0.8, -0.8, 0.8, -0.8],
])
E3 = jnp.array([0.0, 0.0, 1.0])


class RewardSettings(NamedTuple):
    position: float
    velocity: float
    attitude: float
    omega: float
    rotor_command: float
    failure_penalty: float
    learning_scale: float = 1.0


REWARD_PROFILES = {
    "legacy": RewardSettings(.02, .001, .6, .0001, .0005, 25.),
    "position_focus_v1": RewardSettings(1., .1, .6, .0001, .0005, 1500., 50.),
}
POSITION_FOCUS_VARIANTS = ("A_full", "B_symmetry", "B_se23", "K_full_desired_thrust_extension",
                           "L_symmetry_desired_thrust_extension", "L_se23_desired_thrust_extension",
                           "B_prime_se23_wrench")

# Direct wrench policies differ in whether they receive desired thrust derivatives.
WRENCH_VARIANTS = {
    "K_full_desired_thrust_extension": True,
    "L_symmetry_desired_thrust_extension": True,
    "L_se23_desired_thrust_extension": True,
    "B_prime_se23_wrench": False,
}

SE23_TRACKING_ORDER = ("vec(E_R)", "E_p", "E_nu", "E_omega", "omega_d")
SE23_TRACKING_SCALE = jnp.array([1.] * 9 + [2.] * 3 + [1.] * 9)


class ObservationConfig(NamedTuple):
    symmetry: bool
    derivatives: bool
    actual_derivatives: bool = False
    se23: bool = False


VARIANTS = {
    "A_full": ObservationConfig(False, False),
    "B_symmetry": ObservationConfig(True, False),
    "C_full_derivatives": ObservationConfig(False, True),
    "D_symmetry_derivatives": ObservationConfig(True, True),
    "E_full_actual_derivatives": ObservationConfig(False, True, True),
    "F_symmetry_actual_derivatives": ObservationConfig(True, True, True),
    "G_symmetry_thrust_rates": ObservationConfig(True, False),
    "H_dynamic_thrust_extension": ObservationConfig(True, False),
    "I_full_double_thrust_extension": ObservationConfig(False, False),
    "J_symmetry_double_thrust_extension": ObservationConfig(True, False),
    "K_full_desired_thrust_extension": ObservationConfig(False, False),
    "L_symmetry_desired_thrust_extension": ObservationConfig(True, False),
    "B_se23": ObservationConfig(True, False, se23=True),
    "L_se23_desired_thrust_extension": ObservationConfig(True, False, se23=True),
    "B_prime_se23_wrench": ObservationConfig(True, False, se23=True),
}
DEFAULT_VARIANTS = tuple(variant for variant, flags in VARIANTS.items() if not flags.se23)


def make_environment(bank, variant, dt=DT, reward_profile="legacy"):
    """Resolve observation variants and the distinct dynamic-extension plant."""
    if reward_profile not in REWARD_PROFILES:
        raise ValueError(f"Unknown reward profile: {reward_profile}")
    if reward_profile != "legacy" and variant not in POSITION_FOCUS_VARIANTS:
        raise ValueError("Position-focused reward is restricted to A, B, K and L")
    flags = VARIANTS[variant]
    if variant in WRENCH_VARIANTS:
        from envs.quadrotor_desired_thrust import DesiredThrustExtension
        env = DesiredThrustExtension(bank, dt=dt, symmetry=flags.symmetry, se23=flags.se23,
                                     include_thrust_derivatives=WRENCH_VARIANTS[variant])
    elif variant in ("I_full_double_thrust_extension", "J_symmetry_double_thrust_extension"):
        from envs.quadrotor_double_thrust import DoubleThrustExtension
        env = DoubleThrustExtension(bank, symmetry=VARIANTS[variant].symmetry, dt=dt)
    elif variant in ("G_symmetry_thrust_rates", "H_dynamic_thrust_extension"):
        from envs.quadrotor_thrust_rate import DynamicThrustExtension, ThrustRateInputs
        cls = ThrustRateInputs if variant == "G_symmetry_thrust_rates" else DynamicThrustExtension
        env = cls(bank, dt=dt)
    else:
        env = SmoothQuadrotor(bank, dt=dt, **flags._asdict())
    settings = REWARD_PROFILES[reward_profile]
    env.reward_profile = reward_profile
    env.reward_settings = settings
    env.learning_reward_scale = settings.learning_scale
    if hasattr(env, "core"):
        env.core.reward_settings = settings
        env.core.reward_profile = reward_profile
        env.core.learning_reward_scale = settings.learning_scale
    if reward_profile != "legacy":
        env.experiment_metadata = {**getattr(env, "experiment_metadata", {}),
                                   "reward": "position-focused weighted Euclidean tracking cost",
                                   "reward_profile": reward_profile, "reward_settings": settings._asdict()}
    return env


class Trajectory(NamedTuple):
    center: jax.Array
    amplitude: jax.Array
    frequency: jax.Array  # radians / second
    phase: jax.Array
    heading: jax.Array  # common world yaw applied to the entire trajectory
    yaw_amplitude: jax.Array
    yaw_frequency: jax.Array
    yaw_phase: jax.Array


class Reference(NamedTuple):
    pos: jax.Array
    vel: jax.Array
    rot: jax.Array
    omega: jax.Array
    action: jax.Array
    acc: jax.Array
    jerk: jax.Array
    snap: jax.Array
    yaw: jax.Array
    yaw_rate: jax.Array
    yaw_acc: jax.Array


class State(NamedTuple):
    pos: jax.Array
    vel: jax.Array
    rot: jax.Array
    omega: jax.Array
    trajectory: jax.Array
    step: jax.Array
    previous_action: jax.Array


class ActualDerivatives(NamedTuple):
    acc: jax.Array
    jerk: jax.Array
    snap: jax.Array
    heading: jax.Array
    yaw_rate: jax.Array
    yaw_acc: jax.Array
    heading_valid: jax.Array


def hat(w):
    x, y, z = w
    return jnp.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(matrix):
    return jnp.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def rotation_exp(w):
    """Rodrigues formula, stable at zero rotation."""
    theta = jnp.linalg.norm(w)
    cross = hat(w)
    return (jnp.eye(3) + jnp.sinc(theta / jnp.pi) * cross
            + 0.5 * jnp.sinc(theta / (2 * jnp.pi)) ** 2 * (cross @ cross))


def yaw_rotation(yaw):
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def normalized_jet(a, da, dda):
    """Value and first two derivatives of a / ||a||."""
    length = jnp.linalg.norm(a)
    unit = a / length
    dlength = unit @ da
    dunit = (da - unit * dlength) / length
    ddlength = (da @ da + a @ dda - dlength ** 2) / length
    ddunit = (dda - unit * ddlength - 2 * dunit * dlength) / length
    return unit, dunit, ddunit


def flat_jet(t, trajectory):
    """Exact sinusoidal derivatives through snap and yaw acceleration."""
    tr = trajectory
    angle = tr.frequency * t + tr.phase
    Q = yaw_rotation(tr.heading)
    derivatives = [Q @ (tr.amplitude * tr.frequency ** k
                       * jnp.sin(angle + k * jnp.pi / 2)) for k in range(5)]
    derivatives[0] = derivatives[0] + tr.center
    yaw_angle = tr.yaw_frequency * t + tr.yaw_phase
    yaw = tr.heading + tr.yaw_amplitude * jnp.sin(yaw_angle)
    yaw_rate = tr.yaw_amplitude * tr.yaw_frequency * jnp.cos(yaw_angle)
    yaw_acc = -tr.yaw_amplitude * tr.yaw_frequency ** 2 * jnp.sin(yaw_angle)
    return (*derivatives, yaw, yaw_rate, yaw_acc)


def reference_at(t, trajectory):
    """Differential-flatness map; no finite differences or clipped references."""
    pos, vel, acc, jerk, snap, yaw, yaw_rate, yaw_acc = flat_jet(t, trajectory)
    specific_force = acc + GRAVITY * E3
    b3, db3, ddb3 = normalized_jet(specific_force, jerk, snap)
    heading = jnp.array([jnp.cos(yaw), jnp.sin(yaw), 0.0])
    dheading = yaw_rate * jnp.array([-jnp.sin(yaw), jnp.cos(yaw), 0.0])
    ddheading = (yaw_acc * jnp.array([-jnp.sin(yaw), jnp.cos(yaw), 0.0])
                 - yaw_rate ** 2 * heading)
    cross = jnp.cross(b3, heading)
    dcross = jnp.cross(db3, heading) + jnp.cross(b3, dheading)
    ddcross = (jnp.cross(ddb3, heading) + 2 * jnp.cross(db3, dheading)
               + jnp.cross(b3, ddheading))
    b2, db2, ddb2 = normalized_jet(cross, dcross, ddcross)
    b1 = jnp.cross(b2, b3)
    db1 = jnp.cross(db2, b3) + jnp.cross(b2, db3)
    ddb1 = jnp.cross(ddb2, b3) + 2 * jnp.cross(db2, db3) + jnp.cross(b2, ddb3)
    R = jnp.stack([b1, b2, b3], axis=1)
    dR = jnp.stack([db1, db2, db3], axis=1)
    ddR = jnp.stack([ddb1, ddb2, ddb3], axis=1)
    W = R.T @ dR
    omega = vee((W - W.T) / 2)
    dW = R.T @ ddR - W @ W
    alpha = vee((dW - dW.T) / 2)
    torque = INERTIA @ alpha + jnp.cross(omega, INERTIA @ omega)
    thrust = MASS * jnp.linalg.norm(specific_force)
    action = jnp.linalg.solve(ALLOCATION, jnp.concatenate([thrust[None], torque]))
    return Reference(pos, vel, R, omega, action, acc, jerk, snap,
                     yaw, yaw_rate, yaw_acc)


def sample_trajectory(key, frequency_scale=1.0):
    keys = jax.random.split(key, 8)
    return Trajectory(
        jax.random.uniform(keys[0], (3,), minval=-2.0, maxval=2.0),
        jax.random.uniform(keys[1], (3,), minval=0.2, maxval=0.7),
        frequency_scale * jax.random.uniform(keys[2], (3,), minval=0.3, maxval=0.7),
        jax.random.uniform(keys[3], (3,), minval=-jnp.pi, maxval=jnp.pi),
        jax.random.uniform(keys[4], (), minval=-jnp.pi, maxval=jnp.pi),
        jax.random.uniform(keys[5], (), minval=0.15, maxval=0.4),
        frequency_scale * jax.random.uniform(keys[6], (), minval=0.3, maxval=0.6),
        jax.random.uniform(keys[7], (), minval=-jnp.pi, maxval=jnp.pi),
    )


def make_bank(seed, count, steps, dt=DT, frequency_scale=1.0):
    """Fixed reference bank shared by all policies. Fail if a reference is infeasible."""
    if count < 1 or steps < 1 or dt <= 0:
        raise ValueError("Bank size, horizon and dt must be positive")
    keys = jax.random.split(jax.random.PRNGKey(seed), count)
    trajectories = jax.vmap(lambda key: sample_trajectory(key, frequency_scale))(keys)
    times = jnp.arange(steps + 1) * dt
    build = jax.jit(jax.vmap(lambda tr: jax.vmap(reference_at, in_axes=(0, None))(times, tr)))
    bank = jax.block_until_ready(build(trajectories))
    commands = np.asarray(bank.action)
    if not all(np.isfinite(np.asarray(x)).all() for x in bank):
        raise ValueError("Reference generator produced non-finite values")
    if commands.min() < 0 or commands.max() > 1:
        raise ValueError(f"Infeasible reference commands: [{commands.min()}, {commands.max()}]")
    return trajectories, bank


def at(bank, trajectory, step):
    return jax.tree_util.tree_map(lambda x: x[trajectory, step], bank)


def actual_derivative_features(state):
    """Causal model derivatives under the last held rotor commands.

    These are left-limit, within-hold derivatives: thrust rate and acceleration
    are zero. Jerk/snap impulses at command switches are not represented. Nothing
    depends on the policy's next command or on future simulator states.
    """
    wrench = ALLOCATION @ state.previous_action
    thrust, torque = wrench[0], wrench[1:]
    W = hat(state.omega)
    alpha = jnp.linalg.solve(INERTIA, torque - jnp.cross(state.omega, INERTIA @ state.omega))
    dR = state.rot @ W
    ddR = state.rot @ (hat(alpha) + W @ W)
    acc = thrust / MASS * state.rot[:, 2] - GRAVITY * E3
    jerk = thrust / MASS * dR[:, 2]
    snap = thrust / MASS * ddR[:, 2]

    # Invert the reference's horizontal-heading construction using body y.
    # For upright poses h = normalize(b2 x e3). This matches the reference
    # heading convention, unlike identifying yaw rate with body omega_z.
    h = jnp.array([state.rot[1, 1], -state.rot[0, 1], 0.])
    dh = jnp.array([dR[1, 1], -dR[0, 1], 0.])
    ddh = jnp.array([ddR[1, 1], -ddR[0, 1], 0.])
    radius_sq = h @ h
    valid = (radius_sq > 1e-4) & (state.rot[2, 2] > 1e-3)
    denominator = jnp.maximum(radius_sq, 1e-4)
    cross = h[0] * dh[1] - h[1] * dh[0]
    yaw_rate = cross / denominator
    yaw_acc = ((h[0] * ddh[1] - h[1] * ddh[0]) / denominator
               - 2 * (h @ dh) * cross / denominator ** 2)
    heading = jnp.where(valid, h / jnp.sqrt(denominator), jnp.zeros(3))
    return ActualDerivatives(acc, jerk, snap, heading,
                             jnp.where(valid, yaw_rate, 0.),
                             jnp.where(valid, yaw_acc, 0.), valid.astype(jnp.float32))


def se23_tracking_features(state, ref):
    """Extended-pose error, with velocity and rate errors in the actual body.

    Simulator velocities are spatial. Position error is in the reference
    body, and E_R maps actual body coordinates into the reference body.
    omega_d is reference context for the rotating-frame error dynamics.
    """
    rotation_error = ref.rot.T @ state.rot
    position_error = ref.rot.T @ (state.pos - ref.pos)
    velocity_error = state.rot.T @ (state.vel - ref.vel)
    angular_error = state.omega - rotation_error.T @ ref.omega
    return jnp.concatenate([rotation_error.ravel(), position_error, velocity_error,
                            angular_error, ref.omega])


def observation(state, ref, symmetry, derivatives, actual_derivatives=False):
    if symmetry:
        features = [state.rot.T @ (ref.pos - state.pos),
                    state.rot.T @ state.vel, state.rot.T @ ref.vel,
                    (state.rot.T @ ref.rot).ravel(), state.omega, ref.omega,
                    ref.action, state.rot.T @ E3]
    else:
        features = [state.pos, state.vel, state.rot.ravel(), state.omega,
                    ref.pos, ref.vel, ref.rot.ravel(), ref.omega, ref.action]
    if derivatives:
        heading = jnp.array([jnp.cos(ref.yaw), jnp.sin(ref.yaw), 0.0])
        if symmetry:
            features.extend([state.rot.T @ ref.acc, state.rot.T @ ref.jerk,
                             state.rot.T @ ref.snap, state.rot.T @ heading])
        else:
            features.extend([ref.acc, ref.jerk, ref.snap, heading])
        features.append(jnp.stack([ref.yaw_rate, ref.yaw_acc]))
    if actual_derivatives:
        actual = actual_derivative_features(state)
        vectors = [actual.acc, actual.jerk, actual.snap, actual.heading]
        features.extend([state.rot.T @ vector for vector in vectors] if symmetry else vectors)
        features.append(jnp.stack([actual.yaw_rate, actual.yaw_acc, actual.heading_valid]))
    return jnp.concatenate(features)


def observation_scale(symmetry, derivatives, actual_derivatives=False):
    """Fixed isotropic vector scales; data-dependent axis scales would break symmetry."""
    if symmetry:
        scale = [2.] * 3 + [1.] * 3 + [1.] * 3 + [1.] * 9 + [1.] * 3 + [1.] * 3 + [1.] * 4 + [1.] * 3
    else:
        scale = [2.] * 3 + [1.] * 3 + [1.] * 9 + [1.] * 3 + [2.] * 3 + [1.] * 3 + [1.] * 9 + [1.] * 3 + [1.] * 4
    if derivatives:
        scale += [1.] * 14
    if actual_derivatives:
        scale += [10.] * 3 + [20.] * 3 + [50.] * 3 + [1.] * 3 + [5., 10., 1.]
    return jnp.array(scale)


def attitude_error(R, Rd):
    cosine = jnp.clip((jnp.trace(R.T @ Rd) - 1) / 2, -1., 1.)
    return jnp.arccos(cosine)


def advance(state, action, dt=DT):
    """Second-order midpoint integration with an SO(3) attitude update."""
    thrust, torque = (ALLOCATION @ action)[0], (ALLOCATION @ action)[1:]
    def angular_acc(w):
        return jnp.linalg.solve(INERTIA, torque - jnp.cross(w, INERTIA @ w))
    alpha = angular_acc(state.omega)
    wmid = state.omega + dt / 2 * alpha
    Rmid = state.rot @ rotation_exp(wmid * dt / 2)
    acc_mid = Rmid @ (thrust / MASS * E3) - GRAVITY * E3
    return state._replace(pos=state.pos + state.vel * dt + acc_mid * dt ** 2 / 2,
                          vel=state.vel + acc_mid * dt,
                          rot=state.rot @ rotation_exp(wmid * dt),
                          omega=state.omega + dt * angular_acc(wmid),
                          previous_action=action, step=state.step + 1)


class SmoothQuadrotor:
    def __init__(self, bank, symmetry=False, derivatives=False, dt=DT, actual_derivatives=False, se23=False):
        self.bank = bank
        self.symmetry = symmetry
        self.derivatives = derivatives
        self.actual_derivatives = actual_derivatives
        self.se23 = se23
        self.reward_profile = "legacy"
        self.reward_settings = REWARD_PROFILES["legacy"]
        self.learning_reward_scale = 1.0
        self.dt = dt
        self.horizon = bank.pos.shape[1] - 1
        if se23:
            if not symmetry or derivatives or actual_derivatives:
                raise ValueError("SE_2(3) features require symmetry without flat-output derivative features")
            order = (*SE23_TRACKING_ORDER, "u_d")
            self.scale = jnp.concatenate([SE23_TRACKING_SCALE, jnp.ones(4)])
            self.feature_definition = "[" + ",".join(order) + "]"
            self.experiment_metadata = {"symmetry": True, "error_group": "SE_2(3)",
                                        "observation_order": order}
        else:
            self.scale = observation_scale(symmetry, derivatives, actual_derivatives)
        self.obs_size = len(self.scale)

    def observe(self, state):
        ref = at(self.bank, state.trajectory, state.step)
        if self.se23:
            return jnp.concatenate([se23_tracking_features(state, ref), ref.action]) / self.scale
        return observation(state, ref, self.symmetry, self.derivatives, self.actual_derivatives) / self.scale

    def reset(self, key, trajectory=None):
        keys = jax.random.split(key, 5)
        if trajectory is None:
            trajectory = jax.random.randint(keys[0], (), 0, self.bank.pos.shape[0])
        ref = at(self.bank, trajectory, 0)
        # Perturbations in body coordinates preserve the joint yaw symmetry.
        pos = ref.pos + ref.rot @ (0.15 * jax.random.normal(keys[1], (3,)))
        vel = ref.vel + ref.rot @ (0.1 * jax.random.normal(keys[2], (3,)))
        rot = ref.rot @ rotation_exp(0.10 * jax.random.normal(keys[3], (3,)))
        omega = ref.omega + 0.05 * jax.random.normal(keys[4], (3,))
        state = State(pos, vel, rot, omega, jnp.asarray(trajectory), jnp.array(0), ref.action)
        return state, self.observe(state)

    def transition(self, state, action):
        """No auto-reset: reward/terminal observations always precede reset."""
        action = jnp.clip(action, 0., 1.)
        ref = at(self.bank, state.trajectory, state.step)
        next_state = advance(state, action, self.dt)
        next_ref = at(self.bank, state.trajectory, next_state.step)
        ep = jnp.linalg.norm(next_state.pos - next_ref.pos)
        ev = jnp.linalg.norm(next_state.vel - next_ref.vel)
        er = attitude_error(next_state.rot, next_ref.rot)
        ew = jnp.linalg.norm(next_state.omega - next_ref.omega)
        finite = jnp.all(jnp.isfinite(jnp.concatenate([
            next_state.pos, next_state.vel, next_state.rot.ravel(), next_state.omega])))
        failed = (~finite) | (ep > 5.) | (ev > 8.) | (er > jnp.pi / 2) | (ew > 10.)
        truncated = next_state.step >= self.horizon
        q = self.reward_settings
        position_cost = q.position * ep
        velocity_cost = q.velocity * ev
        attitude_cost = q.attitude * er
        omega_cost = q.omega * ew
        command_cost = q.rotor_command * jnp.linalg.norm(action - ref.action)
        reward = -(position_cost + velocity_cost + attitude_cost + omega_cost + command_cost)
        reward = jnp.where(failed, jnp.nan_to_num(reward, nan=-q.failure_penalty) - q.failure_penalty, reward)
        info = {"position_sq": ep ** 2, "velocity_sq": ev ** 2,
                "attitude_sq": er ** 2, "omega_sq": ew ** 2,
                "saturation": jnp.mean((action < .01) | (action > .99)),
                "command_rate_sq": jnp.sum(((action - state.previous_action) / self.dt) ** 2),
                "position_cost": position_cost, "velocity_cost": velocity_cost,
                "attitude_cost": attitude_cost, "omega_cost": omega_cost,
                "rotor_command_cost": command_cost,
                "failure_cost": jnp.where(failed, q.failure_penalty, 0.),
                "failed": failed, "truncated": truncated}
        return next_state, self.observe(next_state), reward, failed, truncated, info


def transform_state(state, yaw, translation):
    Q = yaw_rotation(yaw)
    return state._replace(pos=Q @ state.pos + translation, vel=Q @ state.vel, rot=Q @ state.rot)


def transform_bank(bank, yaw, translation):
    Q = yaw_rotation(yaw)
    rotate = lambda x: jnp.einsum("ij,...j->...i", Q, x)
    return bank._replace(pos=rotate(bank.pos) + translation, vel=rotate(bank.vel),
                         rot=jnp.einsum("ij,...jk->...ik", Q, bank.rot),
                         acc=rotate(bank.acc), jerk=rotate(bank.jerk), snap=rotate(bank.snap),
                         yaw=bank.yaw + yaw)
