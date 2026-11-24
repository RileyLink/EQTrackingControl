<h1 align="center">
  <b>EQ Tracking Control: Symmetry-Aware Trajectory Tracking Controllers with RL</b><br>
</h1>

<p align="center">
 <img src="https://img.shields.io/badge/python-3.10-blue" />
</p>

This repository contains code for the environment and RL training implementations of the paper, [Leveraging Symmetry to Accelerate Learning of Trajectory Tracking Controllers for Free-Flying Robotic Systems](https://arxiv.org/abs/2409.11238). All environments are written in Jax, with wrappers available for the standard Gymnasium interface. A custom PPO implementation, also written in Jax, is provided. Training takes approximately 4 mins for 10M env steps, running on an Nvidia RTX A5000. 

The `codex/quadrotor-weak-symmetry` branch adds weak SE_2(3) observation reduction to the original quadrotor environments and PPO. Select `--equivariant 2` to train with relative tracking errors and their retained context. The fork's `main` branch preserves the upstream code; the thrust-derivative experiments are on `codex/quadrotor-thrust-experiments`.

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

### Quadrotor observation modes

| `--equivariant` | Observation | Input scalars |
| --- | --- | --- |
| `0` | Full actual and reference states | 40 |
| `1` | Released observation branch, preserved for reproducing the upstream code | 31 |
| `2` | Weak SE_2(3) reduction: relative errors, actual angular velocity, reference rotor commands | 25 |
| `3` | Corrected strong SO(2) with translations: separate body-frame velocities and gravity direction | 31 |

Use modes `2` and `3` for the weak-versus-strong comparison. Mode `1` retains the released world-velocity and `R e3` features, which are not exactly yaw invariant. These mode meanings are specific to quadrotors; particle and Astrobee modes retain their original meanings.

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

## Example Results
<img src="./figures/example_results.png" alt="main_results" width="800"/>

<img src="./figures/example_rollouts.png" alt="main_rollouts" width="800"/>

## Checkpoints
Pretrained models are provided within the `checkpoints` directory, for all systems. These models match the performance reported in the paper, and the individual hyperparameters used for each experiment are contained wihtin the `config.txt` file of the individual experiments. 

These are the upstream checkpoints. Train a new 25-input policy using the weak-symmetry command above; changing an evaluation flag does not change a checkpoint's input dimension.

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
