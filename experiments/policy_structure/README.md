# Quadrotor policy structure experiments

These experiments query a saved actor and compare its applied collective thrust and body moments with the Lee–Leok–McClamroch position controller. The workflow uses the existing JAX environment, flatness planner, fixed observation scales, action allocator, and midpoint simulator. [Lee, Leok, and McClamroch, position-controlled flight mode](https://arxiv.org/abs/1003.2005)

## Run

Run from the repository root with the workspace virtual environment. For a separate installation, use the pinned packages in `requirements-quadrotor.txt`.

```powershell
$py = '..\.venv\Scripts\python.exe'
$checkpoint = 'runs/se23_paper_size_full/L_se23_desired_thrust_extension/seed_0'
$out = 'runs/policy_structure_example'

& $py -m unittest discover -s tests -p test_quadrotor_policy_structure.py -v
& $py -m experiments.policy_structure.data --checkpoint $checkpoint --output "$out/data" --count 12
& $py -m experiments.policy_structure.jacobians --dataset "$out/data" --output "$out/jacobians" --samples-per-group 192
& $py -m experiments.policy_structure.lee_fit --dataset "$out/data" --output "$out/lee_fit"
& $py -m experiments.policy_structure.sparse_fit --dataset "$out/data" --output "$out/sparse_fit" --bootstraps 24
```

Each analysis runs independently against the saved dataset. Use a new output directory for each run. If you relocate a checkpoint, pass `--checkpoint` to the analysis with its new path. The analysis verifies the checkpoint and dataset hashes before use.

Use a smooth-task checkpoint directory containing `config.json` and `policy.msgpack`. The experiments support direct-wrench actors and the four-rotor actors on the same plant, including G with its causal thrust-rate observation. The H, I, and J thrust-integrator variants and the legacy Orbax tasks require separate plant and action semantics.

## Checkpoint interface

The paper-size L checkpoint in the example uses a six-layer, 128-wide leaky-ReLU actor. Its 27 observations are

```text
vec(Rd.T R), Rd.T (p - pd), R.T (v - vd), omega - R.T Rd omega_d,
omega_d, Fd, Fdot_d, Fddot_d, tau_d
```

Rotations are flattened row-major. Every channel is divided by the corresponding checkpoint scale. In this run, the position scale is 2 m, the two thrust-derivative scales are about 0.5134 N/s and 3.2752 N/s², and the desired torque scales are `[0.8, 0.8, 1.6]` N·m. The deterministic actor mean passes through `(tanh(mean) + 1) / 2`. For L, these four channels encode collective thrust from 0 to 1 N and symmetric body torque limits. The existing allocator preserves collective thrust and scales torque into the feasible rotor set. For a four-rotor actor, the action is a vector of rotor commands in `[0, 1]`. The allocation matrix gives the applied wrench in both cases.

## Data and closed-loop comparisons

Each split has its own bank of smooth 3D trajectories and paired vertical rate and curvature trajectories. The test set also includes independent 3D trajectories at 1.3 times the nominal frequencies. Rollouts begin at identical seeded reset states for RL and Lee. Bounded probes offset position, velocity, attitude, and angular velocity around the reference and preserve valid rotations and reference actuator history. The saved arrays include the full pre-action state, reference trajectory and time, state source, scaled actor observation, native action, requested and applied actor wrenches, and requested and applied Lee wrenches.

The closed-loop comparisons use the test bank and paired nominal and stressed starts for each model. The metrics include failure and survival time, tracking RMSE, terminal errors, recovery time, rotor usage and slew, and wrench effort. Recovery is the first period of at least 0.5 s below both 0.20 m position error and 0.15 rad attitude error. Tracking RMSE covers active steps through the first failure, so read it alongside the failure fraction and survival time. Rotor saturation counts commands within 0.01 of either rail. The saved traces and per-trial metrics support comparisons across trajectory families and initial conditions. The plots show the first stressed test trajectory of each family.

## 1. Local Jacobians

The simulator uses `m vdot = f R e3 - mg e3` and `Rdot = R hat(omega)`, with body-to-world `R`. With world-frame errors `ep = p - pd` and `ev = v - vd`, the Lee force is `F = m (ad + g e3) - kp ep - kv ev` and the requested thrust is `F · R e3`. Its ideal thrust gradients are `-kp R e3` and `-kv R e3`. A positive restoring gain in the results is the negative projection of the gradient onto `R e3`. The norm and fraction of the orthogonal component describe departures from this direction.

The actor Jacobian is differentiated through its saved scales, tanh, and actuator map. The observation partials use the ambient observation coordinates, including redundant rotation entries. The physical Jacobian composes those partials with the observation map on a valid state tangent, holding the planner and past actions fixed. The right Cayley retraction on SO(3) has the same first derivative as `R exp(hat(eta))`. The moment comparisons include the right tangent at fixed actual rate, the right tangent at fixed planner angular-rate error, and the derivative with respect to the planner's geometric attitude error where that coordinate is locally invertible. Lee's computed command jet changes with the state under each perturbation.

The allocator-stratified results separate the effect of torque projection from the actor's requested wrench. For a direct-wrench policy, the thrust gain at the tanh midpoint divides out the known relative slope `4 u_f (1 - u_f)` wherever it is nonzero. The result describes variation in the latent thrust sensitivity. Finite-difference files include central errors and one-sided discrepancies at two step sizes. Near a leaky-ReLU or allocator switch, one-sided derivatives can differ.

## 2. Lee-structure fit

The coupled baseline builds `Rc` from `F/||F||` and the planner heading. Its angular velocity and acceleration use analytic force and heading derivatives. The thrust and thrust rate used in that jet respect the collective limit. At a vanishing force or parallel heading, the commanded attitude jet falls back to the planner's. The baseline gains `(kp, kv, kR, kOmega) = (4m, 3.2m, 1.2, 0.65)` are nominal choices for this plant and can be set when generating data.

Least squares fits the applied actor wrench after subtracting the ideal feedforward. It uses the same nominal `Rc, omega_c, alpha_c` construction for both fitted models. Each rollout evaluates that construction on its own state using the nominal translation gains. The scalar fit shares the position and velocity gains across axes, and shares the attitude and rate gains across axes. The diagonal fit allows thrust projections in planner body axes and moment feedback in actual body axes. The saved ranks, singular values, and condition numbers describe the linear fits. Fitted gains are unconstrained. These models test a specified linear family with a fixed command construction. Refitting the fully coupled command generator would be a nonlinear problem.

The action metrics include requested and applied predictions, per-channel errors in N or N·m, and errors divided by the one-sided action scales `[1, 0.8, 0.8, 1.6]`. Applied comparisons and all surrogate rollouts use the same thrust-preserving allocator.

## 3. Sparse identification

The three nested libraries contain low-order error polynomials, then geometric and feedforward terms, then nonlinear interactions. The fit uses sequentially thresholded least squares with a small ridge term on RMS-scaled columns and scaled outputs. It selects each output's sparsest threshold within the specified validation error allowance. The representative library for each mode is the sparsest within that allowance of the best aggregate applied validation RMSE. Each fitted library and mode receives its own test scores and closed-loop trials. [Brunton, Proctor, and Kutz](https://doi.org/10.1073/pnas.1517384113)

The direct target is the applied RL wrench. The residual target is the applied RL wrench minus the applied nominal Lee wrench. Residual predictions add the correction to the applied Lee wrench before final allocation. Coefficients in the saved equations multiply unscaled features. Bootstrap selection and sign frequencies come from resampling training trajectories within each family at the selected thresholds. They measure sensitivity to the fit sample. Correlated features and actuator saturation can make individual terms interchangeable.

| Feature name | Definition |
| --- | --- |
| `ep_d`, `ev_d` | Planner-body position and velocity errors `Rd.T ep`, `Rd.T ev` |
| `eR_d`, `ew_d` | Geometric attitude and transported rate errors relative to the planner |
| `eR_command_change`, `ew_command_change` | Computed-command error minus planner error |
| `omega`, `omega_d` | Actual and planned angular velocity in their respective body frames |
| `ep_dot_b3`, `ev_dot_b3` | Error projections onto the current body thrust axis |
| `ep_projection_change`, `ev_projection_change` | Current-axis projection minus planner-axis projection |
| `b3_world_z_minus_1`, `Fd_projection_change` | `b3 · e3 - 1` and `Fd (b3 · b3_d - 1)` |
| `gyro` | `omega × J omega` |
| `transport_d` | `-J (omega × (R.T Rd omega_d))` |
| `alpha_ff_d_minus_tau_d` | `J R.T Rd alpha_d - tau_d` |
| `transport_command_change`, `alpha_ff_command_change` | Change in the transport and angular acceleration terms when using the computed command jet |
| `tau_d`, `Fd_minus_mg`, `Fdot_d`, `Fddot_d` | Desired moment, thrust above hover, and analytic thrust derivatives |

The equations describe predictive associations over the sampled conditions. Use the sensitivity, held-out action, and closed-loop results together when assessing the learned policy.

## Experiments 2–4 using only the L observation

The follow-up uses the desired body’s forward axis for the recovery heading. It reconstructs the desired force, its first two inertial derivatives, and the commanded attitude jet from the saved 27-channel observation. It validates the equations by inserting the original world-horizontal heading and comparing every requested wrench to the saved Lee baseline. The baseline retains the original heading in the rollout comparison.

```powershell
$py = '..\.venv\Scripts\python.exe'
foreach ($seed in 0..2) {
    & $py -m experiments.policy_structure.observation_experiments `
        --dataset "runs/policy_structure_L/seed_$seed/data" `
        --output "runs/policy_structure_observation_L/seed_$seed"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

All models fit the actor's deterministic requested wrench after tanh and before rotor allocation. The reported applied scores project both policies through the existing actuator map. The scalar and axis-specific least-squares fits hold the nominal command construction fixed. The bounded, multistart scalar fit recomputes it at each candidate position and velocity gain.

The sparse fits use the exact terms, a small set of corrections, and a 416-feature library that adds pairwise products and selected higher-order terms. Each library is fit directly and as a correction to the nominal observation-based wrench. `equations.txt` lists every retained term and `sparse_coefficients.csv` records the physical coefficient and selection frequency across training-trajectory resamples. `sparse_thresholds.csv` records the validation fit and term count along each threshold path.

For Experiment 4, `local_gain_metrics.csv` compares constant and state-dependent models of the actor's local derivatives. The moment gains are projections of six attitude-and-rate derivatives onto the two geometric feedback terms for each axis, after subtracting the feedforward derivative. The state-dependent action fits estimate gain functions directly from the wrench and include a constant-gain comparison with the same offset features. The saved gain equations give requested commands, and all fitted controllers are also run through the paired rollout evaluation.
