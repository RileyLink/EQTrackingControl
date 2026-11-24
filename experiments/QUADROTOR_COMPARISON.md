# Smooth-reference quadrotor policy comparison

This compares PPO policies with and without symmetry reduction, explicit desired
flat-output derivatives, and dynamic thrust extensions. A--G use four normalized
rotor-command actions; H uses thrust rate/body torque; I/J use thrust acceleration/body torque;
K/L use direct collective thrust/body torque with only the desired reference extended. It uses
the paper code's mass, inertia and rotor allocation, with a new smooth-reference
task and corrected invariant observations. It is **not** a reproduction of the
paper's random-reference-control training. Legacy scripts/checkpoints are unchanged.

| Variant | Observation | Scalars |
|---|---|---:|
| `A_full` | actual state, desired state, desired rotor commands | 40 |
| `B_symmetry` | invariant state/reference features | 31 |
| `C_full_derivatives` | A plus desired acceleration, jerk, snap, heading, yaw rate and acceleration | 54 |
| `D_symmetry_derivatives` | B plus body-frame desired acceleration, jerk, snap and heading; yaw rate and acceleration | 45 |
| `E_full_actual_derivatives` | C plus actual acceleration, jerk, snap, heading, yaw rate/acceleration, and heading validity | 69 |
| `F_symmetry_actual_derivatives` | D plus the invariant version of E's actual features | 60 |
| `G_symmetry_thrust_rates` | B plus desired and causal interval-averaged actual total-thrust rates | 33 |
| `H_dynamic_thrust_extension` | B plus desired/previous actual thrust rate and the integrated thrust state; thrust-rate/body-torque actions | 34 |
| `I_full_double_thrust_extension` | Full extended state/reference and desired thrust-acceleration/body-torque input | 44 |
| `J_symmetry_double_thrust_extension` | Invariant version of I, with identical extended dynamics | 35 |
| `K_full_desired_thrust_extension` | Actual p/v/R/omega; desired p/v/R/omega/F/Fdot; desired Fddot/body torque; direct f/body-torque actions | 42 |
| `L_symmetry_desired_thrust_extension` | Invariant version of K with the same direct f/body-torque dynamics | 33 |

Matrix rotations use nine coordinates, so these are embedding sizes, not manifold
dimensions. Position/velocity are already in A/B and are not duplicated. Some
heading/acceleration information is redundant with desired attitude/control.
Legend parentheses show `(obs N, act 4)`, meaning network input scalars and
action channels. All policy dimensions and action meanings are also saved in
`policy_dimensions.csv` beside the figures.

## Run in this workspace

From `Learning Symmetry/EQTrackingControl` in PowerShell, use the environment
installed one directory above the repository:

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -p test_quadrotor*.py -v
..\.venv\Scripts\python.exe -m experiments.quadrotor_comparison --preset smoke --output runs/check_pipeline
..\.venv\Scripts\python.exe -m experiments.quadrotor_comparison --preset pilot --seeds 0 1 2 --output runs/pilot
```

For a separate installation, create a venv inside your checkout, install
`requirements-quadrotor.txt`, and use that venv's Python. The legacy requirements
are not needed. The supplied JAX pin supports CPU on this Windows machine; a GPU
machine needs a platform-compatible JAX backend installed per official JAX docs.

All registered variants run by default; `--variants E_full_actual_derivatives F_symmetry_actual_derivatives`
selects a subset. `--seeds 0 1 2` overrides the seeds. `--total-steps`, `--num-envs`,
`--rollout-steps`, `--layers`, `--width`, `--horizon` and `--eval-every` override
presets. Total steps must be a whole number of rollout batches, each divisible by
the minibatch count. Existing output directories are refused.

| Preset | Steps per policy/seed | Default seeds | Network (each actor/critic) | Horizon |
|---|---:|---:|---|---:|
| smoke | 2,048 | 1 | 2 x 32 | 4 s |
| pilot | 262,144 | 1 | 3 x 64 | 20 s |
| research | 999,981,056 | 10 | 6 x 128 | 50 s |

Research rounds one billion steps down to whole batches. Twelve policies and ten
seeds imply nearly 120 billion total transitions. Smoke/pilot reports are marked
**not converged results**. A pilot does not establish a final performance ranking.

## References and frames

State is `(p, v, R, omega)`: `R` maps body to world; `v` is world-frame velocity;
`omega` is body-frame angular velocity. Reference position/heading are analytic
sinusoids with randomized amplitudes, frequencies, phases, translation and yaw.
Derivatives through snap and yaw acceleration are analytic, not finite differences.

Flatness supplies consistent `(p_d, v_d, R_d, omega_d, u_d)` at the same time.
Thrust direction is `(acc_d + g e3) / ||acc_d + g e3||`. A commanded horizontal
heading completes attitude via the usual cross-product construction. This uses a
flatness heading convention; it does not identify yaw rate with `omega_d[2]`.
Analytic derivatives of normalized vectors give `Rdot_d` and `Rddot_d`, then body
angular velocity, torque and rotor allocation. Reference commands are never
clipped: any bank containing non-finite data or sampled commands outside `[0,1]`
is rejected. Position frequencies are 0.3--0.7 rad/s, yaw frequencies 0.3--0.6
rad/s. The faster test scales both by 1.3. Tests independently compare the flatness
map with continuous-time dynamics and automatic differentiation.

The symmetry group is common translation plus yaw, `R^3 semidirect SO(2)`,
isomorphic to `SE(2) x R`. The corrected observation is:

```text
[R.T (p_d-p), R.T v, R.T v_d, vec(R.T R_d), omega, omega_d, u_d, R.T e3]
```

The extra features are:

```text
full:      [acc_d, jerk_d, snap_d, heading_d, yaw_rate_d, yaw_acc_d]
invariant: [R.T acc_d, R.T jerk_d, R.T snap_d, R.T heading_d, yaw_rate_d, yaw_acc_d]
```

`heading_d = [cos(yaw_d), sin(yaw_d), 0]` avoids wrapping and uses the same
three-component embedding for C and D. Rotated derivatives are not derivatives
of vectors expressed in a moving frame.

The released reduced observation retains world velocities and `R[:,2]`; it does
not remove common world yaw. This path corrects those expressions without changing
old policy inputs. Reward norms, termination criteria and reset perturbations also
preserve symmetry. Absolute-position boundaries, Euler-angle subtraction and
per-axis learned observation normalization are not used.

## Training protocol

By default, runs **train only** and their displayed graphs contain only mean
training reward per environment step. Each point averages a PPO rollout batch,
then seeds; bands show seed standard deviation. Evaluation is optional and has
no effect on the training observations or reward.

"Held out" means reference trajectories excluded from the training bank. The
nominal held-out bank uses new amplitudes/frequencies/phases sampled from the same
family, not necessarily harder trajectories. It measures generalization to unseen
references. Repeatedly using that bank for model selection makes it a validation
set; a separate untouched bank would be needed for an unbiased final test.

- All variants share saved train/held-out reference banks, rewards, integration,
  initial-error distributions, PPO settings, and hidden widths/depths. First-layer
  parameter counts differ with input dimension. Generator parameters and trajectory
  index/phase are hidden from all policies. A finite derivative bundle is not
  asserted to be Markov for arbitrary planners: this is an observation/information
  experiment, not a proof of a new quotient MDP.
- Reset errors are sampled in the desired body frame. Reference reset and action
  RNG streams are separate. Evaluation uses identical fixed trajectory indices
  and perturbation keys across all policies/seeds. Training reset times can differ
  when policies fail at different times.
- Dynamics use midpoint integration with an SO(3) exponential update, `dt=0.05`.
  This deliberately improves on the legacy Euler discretization.
- Actor/critic are separate leaky-ReLU MLPs. Gaussian latent actions map through
  `(tanh(z)+1)/2` to the original four rotor commands. Means initialize near hover;
  there is no residual/feedforward controller. PPO likelihood ratios use saved
  pre-tanh samples; the common bijection Jacobian cancels. Optional entropy is
  latent-Gaussian entropy; every comparison preset has zero entropy coefficient.
- Fixed isotropic feature scales divide position vectors by 2 m; remaining unit
  scales are 1 for A--D. E/F additionally scale actual acceleration by 10 m/s^2,
  jerk by 20 m/s^3, snap by 50 m/s^4, heading by 1, yaw rate by 5 rad/s and yaw
  acceleration by 10 rad/s^2; the validity bit is unscaled. Each vector uses one
  isotropic scale. Data-dependent per-axis scaling would break invariance.
- Reward is the negative weighted sum of Euclidean position/velocity error,
  attitude geodesic error, body-angular-velocity error and reference-command error.
  Weights: `0.02, 0.001, 0.6, 0.0001, 0.0005`. Failure adds `-25`; no reach bonus.
  Failure thresholds: 5 m position, 8 m/s velocity, pi/2 attitude, 10 rad/s angular
  velocity, or non-finite state. Reward is computed before reset. Time limits
  bootstrap terminal values; failures do not. GAE stops at either boundary.
- Evaluation uses deterministic actor means at zero steps and fixed intervals,
  without selecting a best seed. Final cases are nominal held-out references,
  common yaw 1.2 rad plus translation `[5,-4,2]` m, and independently sampled
  faster references. B/D should match nominal/transformed results up to numerical
  integration error.
- RMSE is measured up to first failure (including the failing transition), first
  averaged per trajectory, then across trajectories/seeds. Short failed episodes
  can have deceptively low RMSE: interpret with failure fraction and survival time.
  No reset discontinuities enter evaluation tracking metrics.

## Outputs

- `manifest.json`: configuration, device, package versions, seeds, implementation
  and reference-file hashes, and running/complete/failed status.
- `train_references.npz`: generator parameters and reference features at each time.
- When evaluation is requested, `evaluation.csv`: aggregate evaluation curves. `per_trajectory.csv`: final
  individual results for all three cases. `summary.csv`: means/std across seeds.
- `REPORT.md`, `training_reward.png`, `comparison.png`: training reward only.
  `training_summary.csv` records the final update's reward, not a converged score.
  Optional `EVALUATION.md`, `evaluation.png`, `rollouts.png` retain evaluation for later. Shading
  is seed population standard deviation, not a confidence interval; omitted for
  one seed.
- `<variant>/seed_<n>/training.csv`: compact update metrics and elapsed time.
- `policy.msgpack` and `config.json` per seed: latest evaluated policy, saved at
  every evaluation point. Completed runs also save `training_state.msgpack` with
  optimizer/environment/RNG state. There is no automatic resume CLI yet.
- `rollout.npz` per seed: nominal positions/references and active/failure masks.

Regenerate reports without training:

```powershell
..\.venv\Scripts\python.exe -m experiments.quadrotor_comparison --report-only runs/pilot
```

Load a saved actor:

```python
from experiments.quadrotor_comparison import load_policy
from rl.quadrotor_ppo import command

model, params, config = load_policy("runs/pilot/D_symmetry_derivatives/seed_0")
mean, logstd, value = model.apply(params, observation)
normalized_action = command(mean)  # decode with the saved variant's action definition
```

Compare A/B for symmetry, A/C and B/D for derivatives, and their interaction.
Compare C/E and D/F for adding actual derivatives. E/F include **both desired and
actual derivatives**, not actual-only observations.
Later derivative ablations or phase/parameter-aware controls can distinguish
convenient feature engineering from additional reference prediction information.
No ranking of the policies is assumed in advance.

## Actual derivative convention in E/F

Actual features are causal, exact derivatives of the simulator's continuous
rigid-body model under the **last held rotor commands**. They do not depend on
the new action about to be chosen, future states, the desired derivatives, or
finite differences. For held total thrust T and torque tau:

```text
alpha = inv(J) (tau - omega x J omega)
Rdot = R hat(omega)
Rddot = R (hat(alpha) + hat(omega)^2)
actual acceleration = (T/m) R e3 - g e3
actual jerk = (T/m) Rdot e3
actual snap = (T/m) Rddot e3
```

This sets thrust rate and thrust acceleration to zero within a control hold.
It supplies one-sided, within-hold derivatives. Classical jerk/snap at an abrupt
rotor-command switch are not represented, and these features are not noisy sensor
estimates. At reset the virtual held command is initialized to `u_d(0)`; after a
step it is exactly the applied command. Derivative features consequently expose
information about the previous command as well as the current physical state.
Their benefit would reflect model feature engineering and input history, not new
independent physical state information.

Actual heading uses `h = normalize([R[1,1], -R[0,1], 0])`, the inverse of the
reference's heading construction in its upright chart. Heading rate/acceleration
are obtained analytically using Rdot/Rddot. At horizontal body-y projection
squared norm <= 1e-4 or body-z vertical component <= 1e-3, heading/rates are masked
to zero and an extra validity bit is supplied. Acceleration/jerk/snap remain
available. F rotates all four actual vectors by R.T; actual yaw rates and validity
are invariant under a common constant yaw/translation. Actual position/velocity
are already present and are not duplicated.

## Evaluation later

Collect evaluation during a future run with `--evaluate`. It still displays
only training reward by default. To evaluate a completed training-only run later,
without additional PPO training:

```powershell
..\.venv\Scripts\python.exe -m experiments.quadrotor_comparison --evaluate-only runs/pilot
..\.venv\Scripts\python.exe -m experiments.quadrotor_comparison --report-only runs/pilot --include-evaluation-plots
```

The first command loads the saved actors and creates evaluation CSVs and rollout
arrays. The second explicitly renders evaluation to separate files. Historical
four-policy runs remain loadable and can regenerate training-only plots.

## Longer reward-only comparison

```powershell
..\.venv\Scripts\python.exe -m experiments.quadrotor_longer --steps 2097152 --seeds 0 1 2 --output runs/longer_training_reward_2m
```

This uses the same pilot task and network, with a fresh longer learning-rate
schedule. JAX vmap batches independent seeds; each retains separate parameters,
optimizer state, environments and random streams. A batched update was checked
against the corresponding independent update. No held-out evaluations run.
Reward CSVs, policy snapshots at each checkpoint, latest full training states,
and copies of the exact source files are saved. Checkpoint policies enable later
analysis without requiring a new training run. The default 2,097,152 steps per
policy/seed is eight times the earlier pilot budget and remains below paper scale.

## Thrust-rate comparisons G and H

Here F is **scalar collective thrust in N**, not a three-component force vector
and not the name of policy F. Both new policies use the invariant base observation
and remove all explicit acceleration/jerk/snap/yaw-derivative features.

The desired rate is analytic:

```text
F_d = m ||acc_d + g e3||
Fdot_d = m (acc_d + g e3).T jerk_d / ||acc_d + g e3||
```

G retains the exact original rotor-command dynamics and appends just two scalar
features, each scaled by 1 N/s: desired Fdot_d and actual interval-averaged Fdot.
At decision t, the actual feature is `(F_applied[t-1]-F_applied[t-2])/dt`, using
only past applied commands. It initializes to zero. This is a backward difference
across control intervals; the classical within-hold derivative of a constant
rotor command would be zero. It introduces no future-policy-action dependence.

H is a **first-order dynamic extension**:

```text
new state: actual collective thrust F
new actions: nu = Fdot, tau_x, tau_y, tau_z
F_next = clip(F + dt nu, 0, 1 N)
```

H observes the same invariant base, desired Fdot, the previous actually applied
rate, and F. It starts at reference thrust with zero rate. Normalized actor means
start at `[0.5,0.5,0.5,0.5]`, corresponding to zero rate and zero torque. Rate is
bounded to +/-1 N/s and further limited near thrust bounds. Requested body torque
axis limits are `[0.8,0.8,1.6]` Nm. Torque is radially scaled into the rotor-feasible
set over the **entire linear thrust ramp**, preserving collective thrust. The
existing midpoint dynamics use the ramp's midpoint rotor commands. The reward
still evaluates physical tracking and applied midpoint rotor-command error with
the same weights as A--G.

H changes the action formulation and actuator evolution, including exploration
in physical units. Its graph line is not a pure observation-feature ablation.
It is not the double extension with states `(F,Fdot)` and action `Fddot` sometimes
used for snap feedback linearization. No motor lag is added to body-torque actuation.

Train only the new cases at the saved comparison's budget:

```powershell
..\.venv\Scripts\python.exe -m experiments.quadrotor_longer --variants G_symmetry_thrust_rates H_dynamic_thrust_extension --steps 2097152 --seeds 0 1 2 --output runs/thrust_rate_cases_2m
```

Then add them to the saved six lines without retraining or editing those runs:

```powershell
..\.venv\Scripts\python.exe -m experiments.merge_reward_runs --runs runs/longer_training_reward_2m runs/thrust_rate_cases_2m --output runs/longer_training_reward_with_thrust_rates
```

The merger verifies equal PPO settings, budgets, seeds, horizons and reference
bank hashes. It copies raw histories/checkpoints and archives source provenance
from both runs in a fresh output directory. The expanded plot uses the same
16-update smoothing and seed standard-deviation bands as the original graph.

## Second-order thrust extension I and J

Both cases have physical state `(p,v,R,omega,F,Fdot)` and actions
`(nu=Fddot,tau_x,tau_y,tau_z)`. Fdot is the current continuous state, not the
previous action or a finite-difference estimate. Only observations differ:

```text
I: [p,v,vec(R),omega, p_d,v_d,vec(R_d),omega_d, Fddot_d,tau_d, F,Fdot,F_d,Fdot_d]
J: [R.T(p_d-p),R.T v,R.T v_d,vec(R.T R_d),omega,omega_d,
    Fddot_d,tau_d,R.T e3, F,Fdot,F_d,Fdot_d]
```

The desired input `(Fddot_d,tau_d)` replaces the old four-rotor reference in the
observation. Body torque and scalar thrust quantities are invariant under the
common world yaw/translation. The four appended state/reference features have
scales `[1 N,1 N/s,1 N,1 N/s]`; desired input scales are
`[10 N/s^2,0.8 Nm,0.8 Nm,1.6 Nm]`. The physical tracking reward and midpoint
rotor-command cost keep their original weights, with no extra F/Fdot penalties.

The current planner uses analytic sinusoidal position trajectories, followed by
a fixed common yaw rotation Q:

```text
p_d(t) = center + Q [A_i sin(Omega_i t + phase_i)]
j_d(t) = Q [-A_i Omega_i^3 cos(Omega_i t + phase_i)]
s_d(t) = Q [ A_i Omega_i^4 sin(Omega_i t + phase_i)]
q = acc_d + g e3
F_d = m ||q||
Fdot_d = m q.T j_d / ||q||
Fddot_d = m [(j_d.T j_d + q.T s_d)/||q|| - (q.T j_d)^2/||q||^3]
```

Acceleration, jerk and snap are already stored in the shared reference bank.
These formulas do not differentiate actual flight measurements. No explicit
position acceleration/jerk/snap or yaw-derivative channels are added to I/J.
Using a different planner requires its analytic/autodifferentiated derivatives
or a sufficiently smooth fitted reference; sampled positions alone do not supply
exact jerk and snap. Tests compare the thrust formulas with automatic derivatives
of the analytic F_d(t).

The actor initially has zero mean acceleration and torque. Reset uses
`F=F_d(0), Fdot=Fdot_d(0)` and the same paired rigid-body perturbations as the
other runs. Requested acceleration is limited to +/-10 N/s^2, rate to +/-1 N/s,
and collective thrust to `[0,1] N`. Constraints project acceleration into an
admissible interval before integration, preserving enough distance to brake at
later control times. They do not reset F or Fdot. For a held applied nu:

```text
F_next = F + dt Fdot + 0.5 dt^2 nu
Fdot_next = Fdot + dt nu
F_mid = F + 0.5 dt Fdot + 0.125 dt^2 nu
```

The existing SO(3) midpoint rigid-body integrator uses this actual midpoint
thrust. Body torque is radially scaled to keep rotors feasible for the entire
quadratic ramp, including its interior extremum if the thrust rate changes sign.
The two thrust states integrate exactly for a constant nu; the coupled rigid-body
update remains a second-order numerical approximation. No motor lag is added.
The acceleration limit is an explicit modeling choice, available as a constructor
parameter and recorded with the source and configuration; it is not a parameter
claimed from the paper. I/J isolate symmetry within the same extended plant;
comparisons against A--H also change action dynamics and observation information.

Train only the new cases and merge with the saved originals:

```powershell
..\.venv\Scripts\python.exe -m experiments.quadrotor_longer --variants I_full_double_thrust_extension J_symmetry_double_thrust_extension --steps 2097152 --seeds 0 1 2 --output runs/double_thrust_cases_2m
..\.venv\Scripts\python.exe -m experiments.merge_reward_runs --runs runs/longer_training_reward_2m runs/thrust_rate_cases_2m runs/double_thrust_cases_2m --output runs/longer_training_reward_with_double_thrust
```

This adds two lines to the training-reward graph without retraining A--H. Each
new policy retains its CSV history, eight intermediate policy snapshots, final
policy, full optimizer/environment/RNG state, reference archive, and exact source
hashes. Held-out evaluation remains opt-in for later use.

## Actual paths from saved policies

Evaluate completed checkpoints on the same new smooth references, without further
training or altering the saved training-reward data:

```powershell
..\.venv\Scripts\python.exe -m experiments.quadrotor_paths runs/longer_training_reward_with_double_thrust --output runs/path_tracking_2m
```

The default evaluates all saved policy seeds on 12 new 20-second references.
Figures use fixed seed 0 and reference 0; they do not select the best policy seed
or best trajectory. `--seed`, `--trajectory`, `--count` and `--variants` select
different displays or evaluation subsets. Actions use deterministic actor means.
The same reference and physical starting perturbations are paired across cases.
Each actuator model retains its own trained F/Fdot initialization convention.

Outputs include all-policy 3D paths, Euclidean position error over time, a detailed
I/J comparison with x/y/z traces, per-trajectory and aggregate CSVs, and complete
state traces in each policy/seed's `rollout.npz`. Position RMSE uses samples after
the initial point through the first failing transition; no reset or post-failure
frozen samples enter the statistics. Failures can arise from position, velocity,
attitude or angular-velocity limits. Interpret short failed rollouts with their
completion counts and survival time. Each 3D panel uses its own axis limits.

Reference parameters, checkpoints' hashes, raw state histories and exact
evaluation/plot source are retained. To regenerate figures from the saved traces:

```powershell
..\.venv\Scripts\python.exe -m experiments.quadrotor_paths runs/path_tracking_2m --plot-only
```

## Desired-only thrust extension K

K keeps the actual physical state `(p,v,R,omega)` and extends only the desired
reference. Its exact full, world-frame observation is:

```text
[p,v,vec(R),omega, p_d,v_d,vec(R_d),omega_d, F_d,Fdot_d,Fddot_d,tau_d]
```

There are 42 scalars. The first 36 retain the full-state scales of A; the desired
force/rate/acceleration/torque scales are `[1 N,1 N/s,10 N/s^2,0.8 Nm,0.8 Nm,1.6 Nm]`.
These derivative scales are observation normalization, not physical rate limits.
Desired F/Fdot/Fddot use the same analytic planner jet as I/J, satisfying the
desired derivative relationships without numerically integrating reference
states. Neither actual F/Fdot nor separate acceleration/jerk/snap/yaw-derivative
channels are observed.

Four normalized actor outputs encode actual collective thrust `f` in `[0,1] N`
and three body torques with axis limits `[0.8,0.8,1.6] Nm`. Requested torque is
radially scaled into the rotor-feasible set while preserving collective thrust.
The requested thrust is held constant during the current integration interval
and can change at the next decision. There is no actual thrust/rate integrator,
rate limit, acceleration limit or added feedforward controller. The initial
actor mean is hover collective thrust and zero body torque. The original rigid-
body dynamics, reward, reference bank, reset distribution and PPO settings are
otherwise retained. The action parameterization differs from A's rotor commands.

```powershell
..\.venv\Scripts\python.exe -m experiments.quadrotor_longer --variants K_full_desired_thrust_extension --steps 2097152 --seeds 0 1 2 --output runs/desired_thrust_only_2m
..\.venv\Scripts\python.exe -m experiments.quadrotor_paths runs/desired_thrust_only_2m --output runs/path_tracking_desired_thrust_2m
..\.venv\Scripts\python.exe -m experiments.merge_reward_runs --runs runs/longer_training_reward_2m runs/thrust_rate_cases_2m runs/double_thrust_cases_2m runs/desired_thrust_only_2m --output runs/longer_training_reward_with_desired_thrust
```

This trains three independent seeds at the existing 2,097,152-step budget and
tests them on the same 12 new 20-second trajectories and initial conditions used
in `runs/path_tracking_2m`. Existing policies are not retrained or re-evaluated.
The expanded reward graph copies their saved histories into a new output.

### Symmetry version L

L has the same physical dynamics, action mapping, actuator constraints, reward
and desired reference features as K. Only the observation is symmetry-reduced:

```text
[R.T(p_d-p) (3), R.T v (3), R.T v_d (3), vec(R.T R_d) (9),
 omega (3), omega_d (3), R.T e3 (3),
 F_d (1), Fdot_d (1), Fddot_d (1), tau_d (3)]
```

This is 33 inputs and four direct f/body-torque action channels. The physical
state is still `(p,v,R,omega)`. Fixed isotropic scaling preserves the common
world translation/yaw symmetry. K remains the full 42-input counterpart. Their
paired resets and physical transitions are checked for equality, and L's
observations/transitions/reward are checked under a common yaw and translation.

```powershell
..\.venv\Scripts\python.exe -m experiments.quadrotor_longer --variants L_symmetry_desired_thrust_extension --steps 2097152 --seeds 0 1 2 --output runs/desired_thrust_symmetry_2m
..\.venv\Scripts\python.exe -m experiments.quadrotor_paths runs/desired_thrust_symmetry_2m --output runs/path_tracking_desired_thrust_symmetry_2m
..\.venv\Scripts\python.exe -m experiments.merge_reward_runs --runs runs/longer_training_reward_2m runs/thrust_rate_cases_2m runs/double_thrust_cases_2m runs/desired_thrust_only_2m runs/desired_thrust_symmetry_2m --output runs/longer_training_reward_all_12
```

The default plots now label every policy with its observation/action dimensions,
including the original cases. Old source data and checkpoints remain unchanged;
the expanded comparison copies the earlier histories without retraining them.

## Stronger position penalty: A/B/K/L only

The `position_focus_v1` reward profile changes relative tracking priorities:

| Cost term | Legacy weight | Position-focused weight |
|---|---:|---:|
| Euclidean position error [m] | 0.02 | 1.0 |
| Euclidean velocity error [m/s] | 0.001 | 0.1 |
| Attitude error [rad] | 0.6 | 0.6 |
| Angular-velocity error [rad/s] | 0.0001 | 0.0001 |
| Applied rotor-command error | 0.0005 | 0.0005 |
| One-time failure cost | 25 | 1500 |

The reward is the negative sum of these costs. A 30 cm position error contributes
-0.30 per step, versus -0.006 before. Position and velocity costs increase 50x and
100x relative to attitude. No observation, physical dynamics, action mapping,
reference bank, reset distribution or termination threshold is changed.

The larger failure cost accounts for the increased running costs: at gamma 0.995,
the worst-case discounted sum of costs while within all current failure bounds is
less than 1350. A failure costs 1500, preventing early termination from becoming a
cheap substitute for incurring the bounded valid-state running costs.

PPO's critic values and GAE targets use raw rewards divided by 50 for numerical
conditioning. The actor still optimizes the same positively scaled objective;
training reward, episode returns and cost-component logs remain in raw units.
Tests compare this update with explicitly rescaled environment rewards. New CSVs
also record raw position/velocity/attitude RMSE and each reward component.

The factory default remains `legacy` so all existing checkpoints can be
re-evaluated with their original reward. A saved run's profile is propagated into
later evaluation automatically. Mixing different profiles in a merged reward
graph is rejected. The new profile is restricted to A/B/K/L.

```powershell
..\.venv\Scripts\python.exe -m experiments.quadrotor_longer --variants A_full B_symmetry K_full_desired_thrust_extension L_symmetry_desired_thrust_extension --reward-profile position_focus_v1 --steps 2097152 --seeds 0 1 2 --output runs/position_focus_abkl_2m
..\.venv\Scripts\python.exe -m experiments.quadrotor_paths runs/position_focus_abkl_2m --output runs/path_tracking_position_focus_abkl_2m
..\.venv\Scripts\python.exe -m experiments.plot_reward_rmse --run runs/position_focus_abkl_2m --output runs/reward_rmse_position_focus_abkl
..\.venv\Scripts\python.exe -m experiments.compare_reward_profiles --output runs/position_focus_abkl_comparison
..\.venv\Scripts\python.exe -m experiments.compare_quadrotor_paths --evaluation-run runs/path_tracking_position_focus_abkl_2m --output runs/path_comparison_position_focus_abkl
```

These are fresh, paired three-seed runs with the same 2,097,152-step budget as the
original comparison. Only A/B/K/L are trained and evaluated. All original data is
retained. Compare RMSE, completion and attitude across objectives; raw reward
numbers have different meanings and should not be compared as performance scores.

## Extended-pose error coordinates for B and L

`B_se23` and `L_se23_desired_thrust_extension` use the following state inputs.
Here `v` and `v_d` are spatial velocities, and `omega` and `omega_d` are body rates.

| Input | Definition | Frame | Scalars |
|---|---|---|---:|
| `E_R` | `R_d.T @ R` | Actual body to desired body | 9 |
| `E_p` | `R_d.T @ (p - p_d)` | Desired body | 3 |
| `E_nu` | `R.T @ (v - v_d)` | Actual body | 3 |
| `E_omega` | `omega - E_R.T @ omega_d` | Actual body | 3 |
| `omega_d` | Desired angular velocity | Desired body | 3 |

B appends the four desired rotor commands. L appends `F_d`, `Fdot_d`, `Fddot_d`
and the three desired body torques. B has 25 inputs and four rotor actions.
L has 27 inputs and four thrust/torque actions. Each uses the existing
plant, reference bank, reset distribution, action decoding and reward profile.
The separate variant names identify the observation definitions in checkpoints.

All four errors are invariant under the simultaneous `SE_2(3)` action
`(R, p, v) -> (Q R, Q p + a, Q v + b)` on the actual and reference states.
The reference rate is needed because `E_p_dot = E_R @ E_nu - omega_d x E_p`
and `omega = E_omega + E_R.T @ omega_d`. Gravity cancels from the continuous
relative acceleration. The error dynamics use the desired body wrench and rate
as time-varying inputs. The fixed-gravity absolute plant and the hidden,
precomputed reference bank require separate checks for an exact MDP quotient.
The invariance test rotates both motions and adds the common acceleration
`g (Q e3 - e3)` to keep the transformed motions feasible at fixed world gravity.

```powershell
..\.venv\Scripts\python.exe -m experiments.quadrotor_longer --variants B_se23 L_se23_desired_thrust_extension --reward-profile position_focus_v1 --steps 2097152 --seeds 0 1 2 --output runs/se23_bl_2m
..\.venv\Scripts\python.exe -m experiments.quadrotor_paths runs/se23_bl_2m --output runs/path_tracking_se23_bl_2m
```
