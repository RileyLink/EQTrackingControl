# Reinforcement Learning Implementations

## Contents:
- `eval_policy.py`: Code to evaluate a trained model on potentially *different trajectories* than the model was trained on, and generate animations. 
- `eval_vis.py`: Code to evaluate a trained model and save the trajectories to later visualze with ROS. 
- `models.py`: Implementation of the neural networks with configurable layers, nodes, and activations written in Flax. 
- `train_policy.py`: Training code for all environments using a custom PPO implemented in Jax. 

## Training (`train_policy.py`)

For the corrected eight-policy campaign and B200 support, use [CLUSTER_HANDOFF.md](../CLUSTER_HANDOFF.md) on `codex/paper-rewards-b200`. It uses the last SLURM presets, fixes the attitude coefficient, defines quadrotor clipped ReLU in [0,1], and runs up to eight independent GPU jobs. Install the new `requirements-paper-v2-b200.txt` or CPU counterpart in a separate environment. The original generic commands below still use their own defaults; they are not the paper preset recipe.

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

Modes `2` and `3` use the same position error `R.T @ (p_d - p)`. Weak uses `R.T @ (v_d - v)` in place of the two separate velocities, matching the position error's desired-minus-actual sign and actual body frame. The weak attitude and angular-rate features retain their existing definitions.

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
| Velocity | `R.T @ v`, `R.T @ v_d` | `R.T @ (v_d - v)` |
| Attitude | `vec(R.T @ R_d)` | Same |
| Angular context | `omega`, `omega_d` | Same |
| Reference wrench | Stored `F_d`, `tau_d` | Same |
| Input scalars | 30 | 27 |

Weak replaces the two velocities with their error; it does not append an error while retaining the separate velocities. Both robots use desired-minus-actual position and velocity errors in the actual body frame. Their existing attitude and angular-rate features retain their respective conventions. No thrust derivatives are added.

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

Weak-mode `2` checkpoints trained before the position/velocity convention change require their original encoder or retraining. Their input counts are unchanged, so the saved configuration alone does not detect this mismatch.

Example:
```bash
python -m rl.eval_policy --env-name quadrotor_random_walk --load-path ./checkpoints/quadrotor_weak_seed0/model_final --num-envs 1
```

For the strong policy, use `./checkpoints/quadrotor_strong_seed0/model_final`. If the trainer appended a numeric suffix to an existing experiment name, use that actual checkpoint folder. The evaluator reads `EQUIVARIANT` from the saved configuration, overriding its command-line observation-mode argument.

The legacy evaluator uses a fixed 5,000-step rollout and writes `eval_data.pkl` and per-environment plots beside the checkpoint. Its `--num-timesteps` option does not change that fixed length. V2 restores the saved `OUT_ACTIVATION`; for shared-reference comparisons and adjustable horizons, use the `rl.eval_vis` exporter below.

## ROS rollout export (`eval_vis.py`)

Run `python -m rl.eval_vis --help` from the repository root. This exporter restores trained policies, applies their deterministic mean actions, and records trajectories for ROS playback. It does not train or convert policies. Mode names identify independently trained checkpoints:

| Export label | Quadrotor `EQUIVARIANT` | Astrobee `EQUIVARIANT` | Default checkpoint folder |
| --- | --- | --- | --- |
| `full` | `0` | `0` | `checkpoints/<robot>_baseline` |
| `strong` | `1`, original released reduction | `1`, original released reduction | `checkpoints/<robot>_equivariant` |
| `corrected_strong` | `3` | `6` | `checkpoints/<robot>_strong_seed0` |
| `weak` | `2` | `2` | `checkpoints/<robot>_weak_seed0` |

For a new matched comparison, train modes `0` and `1` with the same PPO options as the strong/weak training commands above, using separate experiment names such as `<robot>_full_seed0` and `<robot>_original_strong_seed0`. All four Astrobee checkpoints must use `--symmetry_type 2` to compare the same body-force dynamics. The exporter verifies each saved mode and network input shape and restores the saved hidden/output activations and selected training-seed index. Changing an evaluation label cannot turn a strong checkpoint into a weak policy.

For example, export four newly trained quadrotor checkpoints:

```bash
python -m rl.eval_vis --env quadrotor --reference random_walk \
  --checkpoint full=checkpoints/quadrotor_full_seed0 \
  --checkpoint strong=checkpoints/quadrotor_original_strong_seed0 \
  --checkpoint corrected_strong=checkpoints/quadrotor_strong_seed0 \
  --checkpoint weak=checkpoints/quadrotor_weak_seed0 \
  --rviz-pair corrected_strong weak --seed 0 --train-seed 0 --num-timesteps 2000
```

For Astrobee, replace `quadrotor` with `astrobee` in that command and its checkpoint paths. A checkpoint path can name the experiment directory or its `model_final` subdirectory. Repeated `--checkpoint` options select exactly those modes; omit them to request all four default folders. `--rviz-pair` selects two requested modes for playback, defaulting to the first and last requested modes. A single checkpoint can also be exported, without updating the pair files.

The four trained checkpoints must exist before running the comparison. Metadata without tensor-data files is insufficient. V2 defines `clipped_relu` explicitly as min(max(z,0),1), since the released Flax symbol is unavailable; its exact historical definition could not be verified. V2 checkpoints record that choice and use the modern CPU profile for evaluation. Earlier weak-mode checkpoints also require their original encoder or retraining after the error-convention change.

Evaluation uses one common initial physical state and one precomputed reference for all policies. The first requested checkpoint supplies the common environment settings, including reset covariance and termination bound; each policy supplies its own mode and network weights. Automatic error/time resets are disabled during export so a drifting policy does not restart on a different reference. Bound violations are recorded instead. The shared reference is generated with the robot's state matched to its reference state, then replayed: this also prevents the released Astrobee random-walk generator's dependence on actual angular velocity from creating a different task for each policy. Training environments and PPO are unchanged.

Reference options:

- `--reference lissajous` is the default, preserving the original exporter task. Quadrotor requires `ref_state_list.npy` of shape `(N, T, 18)` and `ref_u_list.npy` of shape `(N, T, 4)` in the working directory. `--trajectory-index 0` selects one shared trajectory; `--num-timesteps` must not exceed its length. Astrobee and particle Lissajous references are generated internally.
- `--reference random_walk` needs no trajectory-bank files for quadrotor or Astrobee.
- `--reference position` uses the position task. Astrobee position observations have different input dimensions, so use checkpoints trained with those dimensions.

The default 2,000 steps correspond to 100 simulated seconds at `dt=0.05`. Arrays contain post-step samples starting at `t=dt`; `--seed` controls initialization/reference generation and `--train-seed` selects a policy within a saved seed batch. If any policy becomes nonfinite, all outputs are truncated to the same finite prefix and the failure step is recorded.

Files are written under `ros2_ws/src/trajectory_visualization/trajectory_visualization/data/<robot>/`, or `--output-dir PATH`:

| Files | Contents |
| --- | --- |
| `<mode>_pos.npy`, `<mode>_quat.npy` | Each policy's world positions `(T, 3)` and quaternion orientations `(T, 4)`, ordered `x, y, z, w` |
| `ref_pos.npy`, `ref_quat.npy` | The shared reference |
| `<mode>_rollout.npz` | Positions, velocities, rotation matrices, angular rates, shared reference channels, mean policy actions before environment clipping, sample times, and bound-violation flags |
| `baseline_pos.npy`, `baseline_quat.npy` | First mode selected by `--rviz-pair` |
| `equivariant_pos.npy`, `equivariant_quat.npy` | Second mode selected by `--rviz-pair` |
| `metrics.csv` | Tracking-error summary for every exported mode |
| `manifest.json` | Checkpoints/configurations, seeds, timestep, pair labels, numerical-failure steps, and metrics |

Position RMSE is `sqrt(mean(||p - p_d||^2))` in metres; maximum distance is the largest Euclidean position error. Velocity RMSE uses `v - v_d` in metres/second. Attitude RMSE uses the angle of `R_d.T @ R` in degrees. `first_bound_violation_step` is a one-based post-step index, with a blank CSV cell if no violation occurred. Metrics use the same exported horizon for all policies.

Exports replace files with the same names in the selected output folder. Quaternions are saved as `*_quat.npy`, matching the ROS reader; the old exporter incorrectly used `*_rotm.npy` for these quaternion arrays. The ROS player still shows two policies at a time; rerun with a different `--rviz-pair` to select another comparison. See [ROS playback commands](../ros2_ws/src/trajectory_visualization/README.md).

The original particle comparison remains available with `--env particle`, using full mode `0` and its original position/velocity/action reduction `3`. Its player additionally receives `anim_ref.npy`, `anim_baseline.npy`, and `anim_eq.npy`.
