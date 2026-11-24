# Fresh B′ and L training

B′ and L use the same state-error and desired-wrench features, action map, dynamics, reward, and reference distribution. L also observes the desired thrust rate and acceleration. Each run starts with a fresh actor, critic, and optimizer and uses PPO alone. The three seeds within a variant share JIT-compiled functions, with separate parameters, optimizer state, and environment and action streams.

The default run uses six hidden layers of 256 units, three seeds, and 8,388,608 simulator steps per seed. The existing observation scales and position-focused reward are used for both variants.

## Run

Run from the repository root in PowerShell.

~~~powershell
$py = '..\.venv\Scripts\python.exe'
$run = 'runs/se23_fresh_bprime_l_yaw_8m'
& $py -m experiments.quadrotor_scratch --output $run
& $py -m experiments.quadrotor_scratch_evaluate $run
~~~

The training command defaults to 64 references per family, 240 control intervals per reference, and a 0.05 s time step. The held-out bank uses 32 references per family and independent random seeds. Model-only snapshots are exported every 1,048,576 steps per seed, and the most recent full training state is saved for restart. Use `--resume` with the existing output directory after an interrupted run. The restart verifies the training sources and reference bank hashes before restoring the optimizer, policy, environment, and random-number state.

## Reference and start distributions

| Family | Reference | Training starts |
| --- | --- | --- |
| Vertical | Matched pairs with the same initial desired wrench and opposite thrust rate or curvature | Existing nominal perturbations |
| Spatial | Smooth three-dimensional paths with mild heading changes | Existing nominal perturbations |
| Yaw | Hover with smooth heading reversals | Angle only, rate only, or angle and rate in randomly selected directions |
| Combined | Stronger lateral motion with an independently varying heading | Lateral position and velocity errors with tilt and angular-rate errors |

Yaw starts hold the reference position, velocity, and thrust axis fixed. The yaw angle magnitude is 10–45 degrees and the yaw-rate magnitude is 0.2–0.95 rad/s, depending on the start mode. The same perturbation distributions and random-number seeds are used for B′ and L. The training environment draws references uniformly at reset. The number of transitions in a family can differ if episodes fail at different times.

## Evaluation

- `rollout_trials.csv` and `rollout_summary.csv` record position, attitude, yaw, yaw-rate, tilt, rotor effort, saturation, and failure metrics for common starts. Yaw is the signed relative ZYX yaw extracted from the desired-frame rotation error. Yaw rate is the body-z component of the transported angular-rate error. Overshoot measures excursions across the desired yaw for runs starting at least 5 degrees off heading. Yaw recovery requires 0.5 consecutive seconds within 5 degrees and 0.15 rad/s. Errors include steps up to the first stop.
- `vertical_preview.csv` compares first commands on reference pairs and short-horizon vertical tracking, with separate results for thrust rate and curvature. The expected held-thrust gap comes from integrating the vertical velocity over one time step. L is also evaluated with its two thrust-derivative inputs set to zero.
- `lee_fit_errors.csv` and `lee_fit_coefficients.csv` compare both actors' requested wrenches with nominal Lee and with the same fitted exact geometric basis. The fit uses common physical probes and independent fit and test trajectories. Its body-forward heading and full reference jet are computed in the offline evaluation. The two trained actors receive their respective observations.
- `tracking_progress.png` and `lee_agreement_progress.png` show held-out metrics over training. Lee is evaluated on the held-out reference bank with the same simulator and actuator constraints.
