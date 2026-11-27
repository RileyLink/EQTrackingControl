# Eight-policy symmetry training report

- Source repository: https://github.com/RileyLink/EQTrackingControl
- Source branch: `codex/paper-rewards-b200`
- Exact source commit:
- Campaign seed: `0`
- GPU model and count:
- Python/JAX/CUDA-plugin versions:
- Smoke campaign: all 8 verified / details
- Full campaign: all 8 verified / details
- Requested budgets: quadrotor 1B steps; Astrobee 400M steps, each policy
- Campaign: paper v2, attitude weight wired; quadrotor clipped ReLU [0,1]
- Network: quadrotor 6 layers x 128 units; Astrobee 6 layers x 256 units
- GPU model / driver / JAX version:
- Compiled GPU preflight and all eight smoke jobs passed:
- Effective reward values from run_metadata.json:
- Training elapsed time per task:
- Weight archive download link accessible to Riley:
- Archive SHA-256:
- Changes to the prescribed settings, if any:

Attach small configuration/metadata/curve files if helpful. Transfer the complete
weight archive separately using the link above; the report alone cannot restore
a policy. Keep the cluster copies until Riley confirms a successful import.
