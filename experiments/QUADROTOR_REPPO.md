# REPPO for the SE₂(3) quadrotor policies

## Implementation plan

1. Use the fresh B′/L PPO run as the comparison baseline. B′ receives 25 inputs, including the SE₂(3) tracking errors and desired wrench. L receives the same inputs plus desired thrust rate and acceleration. Reuse the saved training and held-out reference banks, the practice resets, the position-focused reward, the direct-wrench action map, and the actor architecture and initialization. Inherit the PPO run's seeds, simulator step budget, rollout and minibatch sizes, Adam learning-rate schedule, discount, and checkpoint cadence.
2. Add a JAX REPPO learner with a separate categorical critic `Q(o, a)`. Inputs are each variant's existing observation and its sampled wrench command. Fit the critic with HL-Gauss cross-entropy on entropy-adjusted TD(λ) targets computed once per rollout. Bootstrap at time limits from the final observation. Stop the return at a failure or an episode reset. Use layer normalization and the paper's latent prediction loss for the critic.
3. Draw fresh reparameterized actions for each actor update and differentiate the critic's expected Q value through those actions. Apply the paper's per-observation KL gate against the rollout policy and tune the entropy and KL multipliers in log space. Evaluate forward KL analytically in Gaussian coordinates because the policy and its frozen copy use the same invertible action transform. Use the entropy of the tanh-squashed Gaussian on `[-1, 1]^4`, with the existing affine map to `[0, 1]^4` for the simulator. See [paper Sections 3.1–3.3 and Appendix F](https://arxiv.org/pdf/2507.11019) and the [authors' JAX implementation](https://github.com/cvoelcker/reppo/blob/main/src/jaxrl/reppo.py).
4. Write REPPO checkpoints to a new run directory. Log raw task reward, critic target range and clipping, entropy, KL, and failures. Save the full learner state for restart and actor-only snapshots for comparison.
5. Compare all four conditions on identical held-out reference/start pairs at matched checkpoints and seeds. Use position RMSE, yaw RMSE, recovery and failure rates, and training return curves. Report REPPO minus PPO for each observation variant. Verify the target math, actor initialization, checkpoint loading, and the comparison workflow with a short paired run before interpreting a full training run.

The initial 151-bin critic range is `[-40, 10]` in scaled return units. A 1,500-point failure penalty is 30 in those units. The log records the fraction of targets outside that range. The KL bound is 0.1 and the entropy target is −0.5 nats per action channel, following the sign used in the authors' implementation. The default critic auxiliary loss weight is 1.0. The paper shows that REPPO is sensitive to rollout batch size, so results with this experiment's 4,096-transition batch apply to that setting. [Batch-size ablation](https://arxiv.org/pdf/2507.11019#page=25)

## Run the matched experiment

Run from the `EQTrackingControl` directory in PowerShell.

~~~powershell
$py = '..\.venv\Scripts\python.exe'
$ppo = 'runs/se23_fresh_bprime_l_yaw_8m'
$reppo = 'runs/se23_fresh_bprime_l_yaw_8m_reppo'
& $py -m experiments.quadrotor_reppo --ppo-run $ppo --output $reppo
& $py -m experiments.quadrotor_reppo_compare $reppo --output 'runs/se23_ppo_vs_reppo'
~~~

The trainer uses fresh parameters and the baseline's saved settings and bank files. The default PPO source above uses 8,388,608 simulator steps per seed and three seeds per policy. To resume interrupted training, pass `--resume --output $reppo` to the trainer. For a short paired run, create a small PPO run with `experiments.quadrotor_scratch` and pass its directory to `--ppo-run`.

The comparison writes a plot, raw trial results, summaries by seed and reference/start family, and `reppo_minus_ppo.csv` in its own directory. The plot uses nominal starts for position RMSE and yaw practice starts for yaw RMSE. Shading shows one standard deviation across seeds. A negative position or yaw RMSE delta favors REPPO. A positive recovery-rate delta favors REPPO. Failure rates and survival time put errors from prematurely ended trajectories in context.
