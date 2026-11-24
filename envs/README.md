# Environment Implementations

## Contents:
- `base_envs.py`: Data structures for all environments, as well as the base implementations for all robots with individual parent classes per robot containing initialization, reward computations, and termination computations. 
- `particle_envs.py`: dynamics of the `Particle` system implemented in the `step()` method, resetting logic implemented in `reset()`, and observation implementations in `_get_obs()`
- `astrobee_envs.py`: dynamics of the `Astrobee` system implemented in the `step()` method, resetting logic implemented in `reset()`, and observation implementations in `_get_obs()`
- `quadrotor_envs.py`: dynamics of the `Quadrotor` system implemented in the `step()` method, resetting logic implemented in `reset()`, and observation implementations in `_get_obs()`
- `registry.py`: Utility for registering environments and fetching them by name
- `wrappers.py`: Usefull wrappers for the environments for logging, converting to `Gymnax` or `Gymnasium` environments etc. 

## Registry
Environments are registered with a string name combining the `{robot}_{setting}`. For example, `particle_random_walk_velocity` corresponds to the `Particle` environment, with the setting `random_walk_velocity`. Environments can be accessed by name through the registry, with keyword arguments passed in:
```
import envs.registry as registry

env = registry.get_env_by_name("particle_random_walk_velocity", equivariant=config["EQUIVARIANT"])
```

## Particle Environments
These environments correspond to a simple robotic system, a point particle. There are various references that can be used to train the policy:

- Constant position: `particle_position`
- Constant velocity: `particle_constant_velocity`
- Random walking position: `particle_random_walk_position`
- Random walking velocity: `particle_random_walk_velocity`
- Random walking acceleration: `particle_random_walk_accel`
- Random Lissajous tracking: `particle_random_lissajous`

Additionally, there are various subgroups of the group action that can be considered for the equivariant observation:

- `equivariant=0`: Baseline, no observation reduction
- `equivariant=1`: $\mathbb{R}^3$ Symmetry on the position components only
- `equivariant=2`: $T\mathbb{R}^3$ Symmetry on position and velocity components
- `equivariant=3`: $T\mathbb{R}^3 \times \mathbb{R}^3$ Symmetry on position, velocity, and actions

## Astrobee Environments
These environments correspond to the space vehicle, Astrobee, which is a fully-actuated system. 

Available environments:
- Random position `astrobee_position`
- Random walking inputs: `astrobee_random_walk`
- Random Lissajous tracking: `astrobee_random_lissajous`

The observation modes on this branch are:

| `equivariant` | Observation | Position inputs | Tracking inputs |
| --- | --- | --- | --- |
| `0` | Original full observation | 36 | 42 |
| `1` | Original reduced observation, unchanged | 18 | 30 |
| `2` | Weak SE_2(3), using body-frame relative velocity | 21 | 27 |
| `6` | Corrected strong SE(3), keeping body-frame velocities separate | 24 | 30 |

New modes `2` and `6` require `symmetry_type=2`, the body-force/direct-product setting. Modes `3` and `4` retain their legacy reference-action offset code and are not used for the new observation comparison.

The common features in the new modes are `R.T @ (p_d - p)`,
`vec(R.T @ R_d)`, `omega`, and `omega_d`. Tracking tasks append the stored
reference force and torque. Strong supplies `R.T @ v` and `R.T @ v_d`
separately; weak substitutes `R.T @ (v_d - v)`. Both linear errors use
desired minus actual in the actual body frame. The other features have
identical values in both modes. Neither mode changes action decoding,
dynamics, reward, reset distribution, or reference generation. The position
task omits the reference-wrench channels because they are absent from its
state. See the [Astrobee training commands](../rl/README.md#astrobee-training).

## Quadrotor Environments
These environments correspond to an underactuated quadrotor. 

Available environments:
- Random position `quadrotor_position`
- Random walking inputs: `quadrotor_random_walk`
- Random Lissajous tracking: `quadrotor_random_lissajous`

On `codex/quadrotor-weak-symmetry`, quadrotor observations are selected with:

| `equivariant` | Observation | Input scalars |
| --- | --- | --- |
| `0` | Full actual and reference state | 40 |
| `1` | Released observation branch, preserved unchanged | 31 |
| `2` | Weak SE_2(3) tracking errors and retained angular/reference context | 25 |
| `3` | Corrected strong yaw/translation observation | 31 |

Mode `2` uses `[omega, E_omega, vec(E_R), e_p, e_v, u_d]`, where
`E_R = R_d.T @ R`, `E_omega = omega - E_R.T @ omega_d`,
`e_p = R.T @ (p_d - p)`, and `e_v = R.T @ (v_d - v)`.
Both linear errors use desired minus actual in the actual body frame, so
the position error matches mode `3` and its derivative is
`e_p_dot = e_v - cross(omega, e_p)`. The weak attitude and angular-rate
definitions, actual angular velocity, and four desired rotor commands `u_d`
are retained.

Mode `3` keeps the two velocities separately in the actual body frame and
includes `R.T @ e3`. Mode `1` retains the released world velocities and
`R e3`, so it is not an exactly yaw-invariant baseline. Use modes `2` and
`3` for the weak-versus-strong observation comparison. All modes share the
same dynamics, resets, rewards, and four rotor actions.

Position and random-walk references are generated internally. The Lissajous
environment loads `ref_state_list.npy` and `ref_u_list.npy` from the working
directory; supply those files before using it. See the
[quadrotor training commands](../rl/README.md#quadrotor-training).

## Wrappers (`wrappers.py`)
Included in this repository are some useful wrappers. One such wrapper is the `LogWrapper()` used in `train_policy.py`. Another wrapper is incuded to convert from [Gymnax](https://github.com/RobertTLange/gymnax/tree/main)-style environments to [Gymnasium](https://github.com/RobertTLange/gymnax/tree/main)-style environements, `GymnaxToGymWrapper()`. This allows the use of numpy-based inputs and outputs so other RL implementations can be evaluated. 

Example:
```
env = PointParticlePosition()
env_gym = GymnaxToGymWrapper(env)

obs, info = env_gym.reset()
print(obs, info)

obs, reward, done, info = env_gym.step(env_gym.action_space.sample())
print(obs, reward, done, info)

```
