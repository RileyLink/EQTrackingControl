# Reinforcement Learning Implementations

## Contents:
- `eval_policy.py`: Code to evaluate a trained model on potentially *different trajectories* than the model was trained on, and generate animations. 
- `eval_vis.py`: Code to evaluate a trained model and save the trajectories to later visualze with ROS. 
- `models.py`: Implementation of the neural networks with configurable layers, nodes, and activations written in Flax. 
- `train_policy.py`: Training code for all environments using a custom PPO implemented in Jax. 

## Training (`train_policy.py`)

Run the module from the repository root, after installing the package and its dependencies. This keeps imports and checkpoint paths consistent. List the available options with:

```bash
python -m rl.train_policy --help
```

Examples:
```bash
python -m rl.train_policy --seed 2024 --equivariant 1 --debug --exp-name PPO_particle_equivariant --env-name particle_random_walk_velocity
python -m rl.train_policy --no-debug --exp-name PPO_particle_no_equivariant_seed_0 --env-name particle_random_walk_velocity
python -m rl.train_policy --exp-name astrobee_baseline --env-name astrobee_random_walk
```

### Quadrotor training

Check out `codex/quadrotor-weak-symmetry` before using the new quadrotor modes. The fork's `main` branch contains the original implementation. The CPU installation recipe is `python -m pip install -r requirements-weak-symmetry.txt`, followed by `python -m pip install -e .`.

| Mode | Quadrotor observation | Inputs |
| --- | --- | --- |
| `--equivariant 0` | Full state and reference | 40 |
| `--equivariant 1` | Released observation branch, unchanged | 31 |
| `--equivariant 2` | Weak SE_2(3) errors with actual angular velocity and reference rotor commands | 25 |
| `--equivariant 3` | Corrected strong yaw/translation reduction | 31 |

The released mode `1` retains world velocities and `R e3`; mode `3` expresses the velocities and world vertical in body coordinates to obtain yaw-invariant observations. Compare modes `2` and `3` with all other training options matched. The quadrotor weak reduction is selected by `--equivariant 2`; `--symmetry_type` does not select it.

Start a weak-invariant policy from scratch:

```bash
python -m rl.train_policy --env-name quadrotor_random_walk --equivariant 2 --exp-name quadrotor_weak_seed0 --seed 0 --num-seeds 1 --total-timesteps 10000000
```

Start a matched strong-invariant policy:

```bash
python -m rl.train_policy --env-name quadrotor_random_walk --equivariant 3 --exp-name quadrotor_strong_seed0 --seed 0 --num-seeds 1 --total-timesteps 10000000
```

These commands share the default three-layer, 64-unit network and original PPO configuration. To use a larger network, append the same `--num-layers 6 --num-nodes 128` options to both commands. Likewise, match the timestep budget, reward options, learning rate, rollout sizes, and activation settings. The input layer has 25 or 31 inputs according to the reduction. The output remains four rotor commands.

`--seed` sets the root random seed. `--num-seeds` controls how many independently initialized policies the trainer runs in parallel. The examples use one policy per command; for multiple seed comparisons, repeat both commands with other `--seed` values and distinct experiment names.

For an installation and checkpoint-saving smoke check, use a 64-transition run:

```bash
python -m rl.train_policy --env-name quadrotor_random_walk --equivariant 2 --exp-name quadrotor_weak_smoke --seed 0 --num-seeds 1 --num-envs 2 --num-steps 16 --total-timesteps 64 --update-epochs 1 --num-minibatches 2 --num-layers 1 --num-nodes 16 --no-anneal-lr
```

This checks execution and saving; 64 transitions do not train a useful controller. A timestep budget is rounded down to full `num-envs * num-steps` batches, and the rollout batch size must divide evenly into `num-minibatches`.

Available quadrotor reference settings:

- `quadrotor_position`: stationary reference, generated internally.
- `quadrotor_random_walk`: dynamically evolved reference with random inputs, generated internally.
- `quadrotor_random_lissajous`: requires `ref_state_list.npy` and `ref_u_list.npy` in the working directory. These files are not bundled in the weak-symmetry checkout.

The training command initializes weights from scratch. It has no `--load-path` option; existing full-state or 31-input checkpoints cannot be converted into a 25-input weak policy by changing an evaluation flag.

For SLURM, place the weak training command in your job script and set its working directory to the repository root. The bundled `rl/slurm_run_exps.sh` contains the original quadrotor mode `0` and `1` presets.

### Astrobee training

On this branch, the corrected strong Astrobee mode is `--equivariant 6` and the weak mode is `--equivariant 2`. Both require `--symmetry_type 2`: the body-frame force model with the direct-product tracking cost. Other symmetry settings retain their original observation modes `0` and `1`; the new encoders reject them rather than silently pairing body-frame features with a different force/cost convention.

```bash
python -m rl.train_policy --env-name astrobee_random_walk --symmetry_type 2 --equivariant 6 --exp-name astrobee_strong_seed0 --seed 0 --num-seeds 1 --total-timesteps 10000000
python -m rl.train_policy --env-name astrobee_random_walk --symmetry_type 2 --equivariant 2 --exp-name astrobee_weak_seed0 --seed 0 --num-seeds 1 --total-timesteps 10000000
```

For the tracking environments, the new observations are:

| Feature | Strong mode `6` | Weak mode `2` |
| --- | --- | --- |
| Position | `R.T @ (p_d - p)` | Same |
| Velocity | `R.T @ v`, `R.T @ v_d` | `R.T @ (v - v_d)` |
| Attitude | `vec(R.T @ R_d)` | Same |
| Angular context | `omega`, `omega_d` | Same |
| Reference wrench | Stored `F_d`, `tau_d` | Same |
| Input scalars | 30 | 27 |

Weak replaces the two velocities with their error; it does not append an error while retaining the separate velocities. The actual-body pose convention and angular-rate context differ from the quadrotor weak vector's ordering, but still give SE_2(3)-invariant tracking coordinates. No thrust derivatives are added.

Both modes produce the original six direct force/torque commands, clipped to `[-1, 1]`. Strong uses mode `6` because the legacy modes `3` and `4` contain reference-action offsets; selecting the new modes introduces no such offset. The original dynamics, reward, reset distribution, reference generators, and PPO update are unchanged. The symmetry checks concern the observation maps and equality of the original task under identical state/action inputs.

These modes also work for `astrobee_random_lissajous` and `astrobee_position`. The position task has no stored reference-wrench channels: its strong/weak input sizes are 24/21, while its original modes retain sizes 36/18. Its new encoders keep both angular rates as context, even when the desired rate is zero. For a quick installation check, use the quadrotor smoke command above with `--env-name astrobee_random_walk --symmetry_type 2` and the appropriate `--equivariant 6` or `2` setting and a distinct experiment name.

For checkpoint evaluation, use:

```bash
python -m rl.eval_policy --env-name astrobee_random_walk --load-path ./checkpoints/astrobee_weak_seed0/model_final --num-envs 1
```

The saved configuration supplies both `EQUIVARIANT` and `SYMMETRY_TYPE` when the evaluator reconstructs the environment. Use the strong checkpoint folder to evaluate the corresponding strong policy.

### Saved Data
By default, a new directory will be created at `./checkpoints/{exp-name}` and this is where the resulting data generated during training will be saved. This includes the configuration dictionary made from the default parameters and CLI args (`config.txt`), training curves (`training_data.npz`), the final model weights (`/model_final/`) and some figures (`episode_returns_shaded.png`, and `terminal_timesteps_summary.png`). By default, if an experiment name already exists in the `./checkpoints/` directory, a numerical index will be appended and incremented to the experiment name. 

## Evaluations (`eval_policy.py`)

After training, evaluate the saved checkpoint from the repository root. The evaluator loads `config.txt` next to `model_final`, including the observation mode and network dimensions. Use the same environment name as the training run.

Example:
```bash
python -m rl.eval_policy --env-name quadrotor_random_walk --load-path ./checkpoints/quadrotor_weak_seed0/model_final --num-envs 1
```

For the strong policy, use `./checkpoints/quadrotor_strong_seed0/model_final`. If the trainer appended a numeric suffix to an existing experiment name, use that actual checkpoint folder. The evaluator reads `EQUIVARIANT` from the saved configuration, overriding its command-line observation-mode argument.

The current evaluator uses a fixed 5,000-step rollout and writes `eval_data.pkl` and per-environment plots beside the checkpoint. Its `--num-timesteps` option does not currently change that fixed rollout length. The examples use the default `hard_tanh` output activation: the legacy evaluator does not restore a different `OUT_ACTIVATION` from the configuration, so evaluation of models trained with other output activations needs a separate evaluator correction.
