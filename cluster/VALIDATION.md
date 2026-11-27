# V2 handoff validation, 2026-11-27

- Linux CPU, Python 3.12, JAX 0.6.2 / Flax 0.10.6 / Orbax 0.11.13: **36 tests passed** (about 200 seconds).
- The end-to-end test trains all eight full-architecture smoke policies for 64 steps each, interrupts/resumes one, verifies the final tensor checksums, packages/transfers/restores all eight, and exports both robots' four-mode rollouts. Smoke budgets do not demonstrate learned tracking quality.
- Resumed PPO chunks match uninterrupted updates within numerical tolerances; compact logging preserves the update results.
- Actual reward tests verify the effective attitude weights 0.6/10, reward +1 at perfect tracking, a -25 error-penalty difference, and clipped-ReLU values/gradients. A misspelled attitude argument fails explicitly.
- CPU preflight output parses as JSON and verifies all eight tasks. Production preflight rejects CPU-only or multiple-visible-GPU allocations. GPU kernels still need validation on the cluster.
- Linux Bash syntax checks pass. Harmless interpreter stubs verify array task 3 and single-allocation worker 7 select the correct policy/output root and preserve their GPU visibility.
- The full Linux NVIDIA requirements resolve without dependency conflicts; CUDA 12.8 package pins were checked against PyPI metadata. This is dependency validation, not a B200 hardware test.
- Windows CPU validation also passed: 35 full-suite tests, then four reward/device-isolation tests including the newly added guard. Windows fixture tests bypass only Orbax's I/O-throughput diagnostic because tiny Windows fixture writes can yield NaN timing statistics; real tensor save/restore still runs. Linux uses no diagnostic bypass.
- Tested Python source files match the final package's training, environment, and test sources. Discrete dynamics, observation encoders, and PPO update equations were not changed for v2.

Not tested locally: B200 execution, scheduler/account/partition availability, actual eight-GPU concurrency, full-budget training performance, or learned tracking success. The friend must run the GPU preflight/smoke array before the full campaign and return its logs. No GPU speedup or paper-level tracking accuracy is asserted from CPU tests.
