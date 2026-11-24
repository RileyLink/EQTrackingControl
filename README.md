<h1 align="center">
  <b>EQ Tracking Control: Symmetry-Aware Trajectory Tracking Controllers with RL</b><br>
</h1>

<p align="center">
 <img src="https://img.shields.io/badge/python-3.10-blue" />
</p>

This repository contains code for the environment and RL training implementations of the paper, [Leveraging Symmetry to Accelerate Learning of Trajectory Tracking Controllers for Free-Flying Robotic Systems](https://arxiv.org/abs/2409.11238). All environments are written in Jax, with wrappers available for the standard Gymnasium interface. A custom PPO implementation, also written in Jax, is provided. Training takes approximately 4 mins for 10M env steps, running on an Nvidia RTX A5000. 

The `codex/quadrotor-weak-symmetry` branch adds corrected strong and weak observation reductions to the original quadrotor and Astrobee environments and PPO. The fork's `main` branch preserves the upstream code; the thrust-derivative experiments are on `codex/quadrotor-thrust-experiments`.

## Installation

The original experiments used Python 3.10. The weak-symmetry CPU training command has also been smoke-tested on Python 3.11. Use an activated virtual environment. The CPU dependency profile below pins the core versions used for that check; the upstream `requirements.txt` is retained separately. GPU/TPU backend installation is separate from this CPU setup.

```bash
git clone --branch codex/quadrotor-weak-symmetry https://github.com/RileyLink/EQTrackingControl.git
cd EQTrackingControl
python -m pip install -r requirements-weak-symmetry.txt
python -m pip install -e .
```

If you would like to use the ROS2 visualizations, ROS2 must be installed seperately. Then, the following line can be used to install the required ROS packages.

```bash
python -m pip install -r requirements_ros2.txt
```

## Running the code

Run these commands from the repository root of the weak-symmetry branch or worktree. The original PPO trainer initializes a new policy for each run.

In the observation tables, $p$ and $v$ are world-frame position and velocity, $R$ maps body coordinates to world coordinates, and $\omega$ is body-frame angular velocity. Subscript $d$ denotes the reference, and $\mathrm{vec}$ flattens a rotation matrix into nine scalar inputs. Features are grouped by quantity; table rows do not specify concatenation order.

### Quadrotor observation modes

This table describes `quadrotor_random_walk` and `quadrotor_random_lissajous`. The numbers in the column headings are the `--equivariant` settings.

| Quantity | Full: `0` | Original reduced: `1` | Corrected strong: `3` | Weak: `2` |
| --- | --- | --- | --- | --- |
| Position | $p, p_d$ | $R^\top(p_d-p)$ | $R^\top(p_d-p)$ | $R^\top(p_d-p)$ |
| Velocity | $v, v_d$ | $v_d, v$ | $R^\top v_d, R^\top v$ | $R^\top(v_d-v)$ |
| Attitude | $\mathrm{vec}(R), \mathrm{vec}(R_d)$ | $\mathrm{vec}(R^\top R_d)$ | $\mathrm{vec}(R^\top R_d)$ | $\mathrm{vec}(R_d^\top R)$ |
| Angular velocity | $\omega, \omega_d$ | $\omega_d, \omega$ | $\omega_d, \omega$ | $\omega, E_\omega$ |
| Reference commands | $u_d$, four scalars | Same | Same | Same |
| Separate vertical-direction input | Recoverable from full $R$ | $Re_3$ | $R^\top e_3$ | None |
| **Total inputs** | **40** | **31** | **31** | **25** |

Here $E_\omega=\omega-R^\top R_d\omega_d$, $u_d$ contains the four reference rotor commands, and $e_3=(0,0,1)^\top$ is world vertical. Counts are scalar network inputs, including the nine entries of each supplied rotation matrix.

Corrected strong and weak use the same position error $e_p=R^\top(p_d-p)$. Weak uses the matching velocity error $e_v=R^\top(v_d-v)$: both are desired minus actual in the actual body frame. Their kinematic relation is $\dot e_p=e_v-\omega\times e_p$. The weak attitude and angular-rate features retain their existing definitions.

For `quadrotor_position`, the original mode `1` uses $R_d^\top(p_d-p)$ and $\mathrm{vec}(R_d^\top R)$ for its position and attitude entries instead. Its other entries and all four input counts are as above; modes `0`, `2`, and `3` use the same encoders across the three quadrotor tasks.

Use modes `2` and `3` for the weak-versus-strong comparison. Mode `1` retains the released world-velocity and $Re_3$ features, which are not exactly yaw invariant. Mode numbers are environment-specific: Astrobee's corrected strong mode is `6`.

### Train a weak-invariant quadrotor policy

```bash
python -m rl.train_policy --env-name quadrotor_random_walk --equivariant 2 --exp-name quadrotor_weak_seed0 --seed 0 --num-seeds 1 --total-timesteps 10000000
```

Train the matched strong-invariant policy with the same options, changing only the observation mode and output name:

```bash
python -m rl.train_policy --env-name quadrotor_random_walk --equivariant 3 --exp-name quadrotor_strong_seed0 --seed 0 --num-seeds 1 --total-timesteps 10000000
```

Both commands use the original default network (three hidden layers of 64 units), PPO settings, rotor actions, rewards, and reset distribution. The network input dimension follows the selected observation. Random-walk references are generated internally, so these commands need no external trajectory files. The requested timestep budget is rounded down to complete rollout batches; these examples specify a starting budget, not a convergence result.

The trainer requires `--exp-name`. Training starts from scratch; `--load-path` is an evaluation argument. For a tiny installation and checkpoint-saving check, matched larger-network options, evaluation instructions, and the Lissajous prerequisites, see [the training guide](rl/README.md#quadrotor-training).

### Astrobee observation modes

This table describes `astrobee_random_walk` and `astrobee_random_lissajous` with `--symmetry_type 2`, which selects body-frame force commands and the original direct-product tracking cost. Mode `6` is corrected strong SE(3); mode `2` is weak SE_2(3).

| Quantity | Full: `0` | Original reduced: `1` | Corrected strong: `6` | Weak: `2` |
| --- | --- | --- | --- | --- |
| Position | $p, p_d$ | $R^\top(p_d-p)$ | $R^\top(p_d-p)$ | $R^\top(p_d-p)$ |
| Velocity | $v, v_d$ | $v, v_d$ | $R^\top v, R^\top v_d$ | $R^\top(v_d-v)$ |
| Attitude | $\mathrm{vec}(R), \mathrm{vec}(R_d)$ | $\mathrm{vec}(R^\top R_d)$ | $\mathrm{vec}(R^\top R_d)$ | $\mathrm{vec}(R^\top R_d)$ |
| Angular velocity | $\omega, \omega_d$ | $\omega, \omega_d$ | $\omega, \omega_d$ | $\omega, \omega_d$ |
| Reference commands | $F_d, \tau_d$, six scalars | Same | Same | Same |
| Separate vertical-direction input | None | None | None | None |
| **Total inputs** | **42** | **30** | **30** | **27** |

The reference force $F_d$ and torque $\tau_d$ each have three components and are supplied as stored by the reference generator. Position and velocity errors use the same actual-body, desired-minus-actual convention as the quadrotor. Astrobee's weak encoder keeps the same pose and angular-rate features as its corrected strong encoder; only the velocity block is replaced. No separate vertical/gravity-direction feature is supplied in any Astrobee mode.

For `astrobee_position`, the full, corrected strong, and weak columns omit the six reference-command scalars, giving **36**, **24**, and **21** inputs. Its original mode `1` instead supplies $[R_d^\top(p-p_d), v-v_d, \mathrm{vec}(R^\top R_d), \omega-\omega_d]$, giving **18** inputs.

The new modes `2` and `6` require `--symmetry_type 2`. In the released mode `1`, settings `0` and `1` instead use the world-frame position error $p_d-p$; settings `2` and `3` use the body-frame position error shown in the tracking table. Mode `6` avoids the legacy action-offset branches under modes `3` and `4`.

### Train strong and weak Astrobee policies

```bash
python -m rl.train_policy --env-name astrobee_random_walk --symmetry_type 2 --equivariant 6 --exp-name astrobee_strong_seed0 --seed 0 --num-seeds 1 --total-timesteps 10000000
python -m rl.train_policy --env-name astrobee_random_walk --symmetry_type 2 --equivariant 2 --exp-name astrobee_weak_seed0 --seed 0 --num-seeds 1 --total-timesteps 10000000
```

The paired commands share their six direct force/torque outputs, dynamics, rewards, resets, reference generators, and PPO implementation. See [the Astrobee training guide](rl/README.md#astrobee-training).

## Example Results
<img src="./figures/example_results.png" alt="main_results" width="800"/>

<img src="./figures/example_rollouts.png" alt="main_rollouts" width="800"/>

## Checkpoints
Pretrained models are provided within the `checkpoints` directory, for all systems. These models match the performance reported in the paper, and the individual hyperparameters used for each experiment are contained wihtin the `config.txt` file of the individual experiments. 

These are the upstream checkpoints. Train a new weak policy using the appropriate command above; changing an evaluation flag does not change a checkpoint's input dimension.

Earlier weak-mode `2` checkpoints used different position/velocity conventions. They require their original encoder or retraining with the current encoder, even though the input counts have not changed. The saved configuration does not distinguish these encoder revisions.

## Citation
If you use this repository or paper please cite the following:
```bibtex
@misc{welde2024leveragingsymmetryacceleratelearning,
  title={Leveraging Symmetry to Accelerate Learning of Trajectory Tracking Controllers for Free-Flying Robotic Systems}, 
  author={Jake Welde and Nishanth Rao and Pratik Kunapuli and Dinesh Jayaraman and Vijay Kumar},
  year={2024},
  eprint={2409.11238},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2409.11238}
}
```
