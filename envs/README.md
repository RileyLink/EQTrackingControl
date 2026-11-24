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

Mode `2` uses `[omega, E_omega, vec(E_R), E_p, E_R E_nu, u_d]`, where
`E_R = R_d.T @ R`, `E_omega = omega - E_R.T @ omega_d`,
`E_p = R_d.T @ (p - p_d)`, and `E_R E_nu = R_d.T @ (v - v_d)`.
The simulator stores world-frame velocity `v`; the last expression converts
its relative velocity to the desired body frame. Actual angular velocity and
the four desired rotor commands `u_d` are retained.

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
