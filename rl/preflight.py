"""Execute device kernels and verify effective rewards before a long campaign."""

import importlib.metadata
import os
import subprocess


JAX_VERSION = "0.6.2"


def check_device(require_gpu=True):
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jl
    import numpy as np
    if jax.__version__ != JAX_VERSION:
        raise ValueError(f"Campaign v2 requires JAX {JAX_VERSION}; install requirements-paper-v2-b200.txt "
                         "or requirements-paper-v2-cpu.txt in a new environment.")
    devices = jax.devices()
    gpu = [device for device in devices if device.platform == "gpu"]
    if require_gpu and len(gpu) != 1:
        raise ValueError(f"Each policy needs exactly one visible GPU, found {len(gpu)}. "
                         "Use one SLURM GPU per array task; CPU fallback is not allowed.")
    if gpu:
        for name in ("jax-cuda12-plugin", "jax-cuda12-pjrt"):
            if importlib.metadata.version(name) != JAX_VERSION:
                raise ValueError(f"{name} must match JAX {JAX_VERSION}.")

    @jax.jit
    def kernels(key):
        matrix = jax.random.normal(key, (64, 64))
        rotation = jl.expm(jnp.array([[0., -.1, 0.], [.1, 0., 0.], [0., 0., 0.]]))
        return jnp.mean(matrix @ matrix.T), rotation @ jnp.linalg.inv(rotation)
    values = jax.block_until_ready(kernels(jax.random.PRNGKey(0)))
    if not all(np.isfinite(np.asarray(value)).all() for value in values):
        raise ValueError("Compiled PRNG/matmul/matrix-exponential device check returned nonfinite results.")
    np.testing.assert_allclose(np.asarray(values[1]), np.eye(3), atol=1e-5)
    details = [dict(platform=device.platform, kind=device.device_kind,
                    compute_capability=str(getattr(device, "compute_capability", "unknown")),
                    device=str(device)) for device in devices]
    driver = None
    if gpu:
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return dict(devices=details, compiled_kernel_check="passed", nvidia_smi=driver,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))


def check_rewards(config):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from rl.train_policy import make_environment
    env = make_environment(config)
    expected = dict(reward_q_pos="REWARD_Q_POS", reward_q_vel="REWARD_Q_VEL",
                    reward_q_rot="REWARD_Q_ROTM", reward_q_omega="REWARD_Q_OMEGA",
                    reward_r="REWARD_R", reach_reward="REWARD_REACH", terminal_reward="TERMINAL_REWARD")
    for attribute, key in expected.items():
        if getattr(env, attribute) != config[key]:
            raise ValueError(f"Effective {attribute}={getattr(env, attribute)} differs from {key}={config[key]}.")
    state, _ = env.reset(jax.random.PRNGKey(11))
    matching = dict(pos=jnp.zeros(3), vel=jnp.zeros(3), rotm=jnp.eye(3), omega=jnp.zeros(3),
                    ref_pos=jnp.zeros(3), ref_vel=jnp.zeros(3), ref_rotm=jnp.eye(3), ref_omega=jnp.zeros(3))
    if "quadrotor" in config["env_name"]:
        command = env.hover_thrust
        matching["ref_action"] = command
    else:
        command = jnp.zeros(6)
        matching.update(ref_F=jnp.zeros(3), ref_tau=jnp.zeros(3))
    state = state.replace(**matching)
    perfect_reward = float(env._get_reward(state, command, command))
    np.testing.assert_allclose(perfect_reward, 1., atol=1e-6)
    failed = state.replace(pos=jnp.array([env.termination_bound + 1., 0., 0.]))
    # The same state/action isolates the configured error penalty.
    baseline = make_environment(config, terminal_reward=0.)
    penalty_difference = float(env._get_reward(failed, command, command) - baseline._get_reward(failed, command, command))
    np.testing.assert_allclose(penalty_difference, -25., atol=1e-5)
    return dict(effective={key: getattr(env, key) for key in expected},
                perfect_tracking_reward=perfect_reward, failure_penalty_difference=penalty_difference,
                observation_inputs=env.observation_space().shape[0])
