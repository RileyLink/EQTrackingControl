"""Compare checkpointed policies on one reference and export RViz rollouts.

Run from the repository root with ``python -m rl.eval_vis --help``.
The checkpoint, rather than an evaluation flag, determines each observation mode.
"""

import argparse
import ast
import csv
from dataclasses import dataclass, fields
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orbax_cp
from scipy.spatial.transform import Rotation

import envs.registry as registry
from rl.models import ActorCritic


MODE_IDS = {
    "quadrotor": {"full": 0, "strong": 1, "corrected_strong": 3, "weak": 2},
    "astrobee": {"full": 0, "strong": 1, "corrected_strong": 6, "weak": 2},
    "particle": {"full": 0, "strong": 3},
}
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Policy:
    label: str
    checkpoint: Path
    config: dict
    env: object
    model: ActorCritic
    params: dict


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=MODE_IDS)
    parser.add_argument("--reference", choices=("lissajous", "random_walk", "position"),
                        default="lissajous", help="Common evaluation task (default: lissajous).")
    parser.add_argument("--checkpoint", action="append", metavar="MODE=PATH",
                        help="Repeat for each policy; MODE is full, strong, corrected_strong, or weak. "
                             "PATH is an experiment folder or its model_final folder. "
                             "Without this option, use the documented default checkpoint folders.")
    parser.add_argument("--rviz-pair", nargs=2, metavar=("BASELINE_MODE", "EQUIVARIANT_MODE"),
                        help="Modes copied to the existing RViz player's two policy slots. "
                             "Defaults to the first and last requested modes.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to the robot's ROS data folder.")
    parser.add_argument("--seed", type=int, default=0, help="Initial-state and reference seed.")
    parser.add_argument("--train-seed", type=int, default=0, help="Seed index within each checkpoint.")
    parser.add_argument("--num-timesteps", type=int, default=2000)
    parser.add_argument("--trajectory-index", type=int, default=0,
                        help="Shared quadrotor Lissajous trajectory-bank index (default: 0).")
    args = parser.parse_args(argv)
    if args.num_timesteps <= 0 or args.train_seed < 0 or args.trajectory_index < 0:
        parser.error("num-timesteps must be positive; train-seed and trajectory-index must be nonnegative")
    if args.env == "particle" and args.reference == "random_walk":
        parser.error("particle export supports lissajous or position references")
    return args


def checkpoint_requests(robot, specs):
    if not specs:
        folders = {"full": f"{robot}_baseline", "strong": f"{robot}_equivariant",
                   "corrected_strong": f"{robot}_strong_seed0", "weak": f"{robot}_weak_seed0"}
        if robot == "particle":
            folders["strong"] = "particle_pva_equivariant"
        return {label: Path("checkpoints") / folders[label] for label in MODE_IDS[robot]}
    requests = {}
    for spec in specs:
        label, separator, path = spec.partition("=")
        if not separator or not path or label not in MODE_IDS[robot] or label in requests:
            raise ValueError(f"Invalid or duplicate checkpoint {spec!r}; use a unique MODE=PATH from "
                             f"{', '.join(MODE_IDS[robot])}.")
        requests[label] = Path(path)
    return requests


def read_checkpoint(path):
    path = path.expanduser().resolve()
    config_path = path / "config.txt"
    checkpoint = path / "model_final"
    if not config_path.is_file():
        config_path, checkpoint = path.parent / "config.txt", path
    if not config_path.is_file() or not checkpoint.is_dir():
        raise FileNotFoundError(f"Missing config.txt or checkpoint for {path}. Train the requested "
                                "mode first, or pass --checkpoint MODE=PATH for available modes.")
    config = ast.literal_eval(config_path.read_text())
    return checkpoint, config


def environment_kwargs(config):
    # Preserve the trainer's constructor arguments; the first policy defines the common task.
    names = ("TERMINATION_BOUND", "TERMINAL_REWARD", "STATE_COV_SCALAR", "REF_COV_SCALAR",
             "REWARD_Q_POS", "REWARD_Q_VEL", "REWARD_Q_ROTM", "REWARD_Q_OMEGA", "REWARD_R",
             "REWARD_REACH", "USE_DES_ACTION_IN_REWARD", "USE_ABS_REWARD_FN", "REWARD_FN_TYPE",
             "CLIP_ACTIONS", "SYMMETRY_TYPE", "DT")
    kwargs = {name.lower(): config[name] for name in names if name in config}
    # Continuous, equal-horizon evaluation: record bound violations instead of resetting.
    kwargs.update(terminate_on_error=False, max_time=float("inf"))
    return kwargs


def select_seed_params(params, seed_index=0):
    if seed_index < 0:
        raise ValueError("Training seed index must be nonnegative.")
    log_std = np.asarray(params["log_std"])
    if log_std.ndim == 1:
        if seed_index != 0:
            raise ValueError("This checkpoint contains one unbatched policy; use --train-seed 0.")
        return params
    if log_std.ndim != 2 or seed_index >= log_std.shape[0]:
        raise ValueError(f"Training seed index {seed_index} is outside this checkpoint's seed batch.")
    return jax.tree_util.tree_map(lambda value: value[seed_index], params)


def restore_params(checkpoint):
    # GPU-trained checkpoints must also be loadable on a CPU evaluation machine.
    checkpointer = orbax_cp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(str(checkpoint))
    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    restore_args = jax.tree_util.tree_map(
        lambda value: (orbax_cp.ArrayRestoreArgs(sharding=sharding)
                       if isinstance(value, orbax_cp.metadata.ArrayMetadata)
                       else orbax_cp.RestoreArgs()), metadata,
    )
    return checkpointer.restore(str(checkpoint), restore_args=restore_args)[0]["params"]["params"]


def load_policies(robot, reference, requests, train_seed):
    sources = [(label, *read_checkpoint(path)) for label, path in requests.items()]
    common_config = sources[0][2]
    env_name = f"{robot}_{'random_lissajous' if reference == 'lissajous' else reference}"
    policies = []
    for label, checkpoint, config in sources:
        expected_mode = MODE_IDS[robot][label]
        if config.get("EQUIVARIANT") != expected_mode:
            raise ValueError(f"{label}: checkpoint has EQUIVARIANT={config.get('EQUIVARIANT')}, "
                             f"expected {expected_mode}; use a checkpoint trained for this mode.")
        trained_env = config.get("env_name", config.get("ENV_NAME", ""))
        if any(other in trained_env for other in MODE_IDS if other != robot):
            raise ValueError(f"{label}: checkpoint belongs to a different robot: {trained_env}.")
        if robot == "astrobee" and config.get("SYMMETRY_TYPE", 0) != common_config.get("SYMMETRY_TYPE", 0):
            raise ValueError("Astrobee checkpoints must share SYMMETRY_TYPE so force commands "
                             "have the same meaning. The four-mode comparison requires type 2.")
        env = registry.get_env_by_name(env_name, equivariant=expected_mode,
                                       **environment_kwargs(common_config))
        model = ActorCritic(
            env.action_space().shape[0], activation=config.get("ACTIVATION", "tanh"),
            num_layers=config.get("NUM_LAYERS", 3), num_nodes=config.get("NUM_NODES", 64),
            out_activation=config.get("OUT_ACTIVATION", "hard_tanh"),
        )
        try:
            template = model.init(jax.random.PRNGKey(0), jnp.zeros(env.observation_space().shape))["params"]
        except AttributeError as error:
            raise ValueError(f"{label}: saved output activation {model.out_activation!r} is unavailable "
                             "in the current ActorCritic/Flax implementation. Restore its original "
                             "activation implementation before evaluating this checkpoint.") from error
        params = select_seed_params(restore_params(checkpoint), train_seed)
        shapes = lambda tree: jax.tree_util.tree_map(lambda value: value.shape, tree)
        if shapes(params) != shapes(template):
            raise ValueError(f"{label}: checkpoint shapes do not match this environment's "
                             f"{env.observation_space().shape[0]} inputs and network configuration.")
        policies.append(Policy(label, checkpoint, config, env, model, params))
    return policies


def initial_state(env, robot, reference, seed, trajectory_index, num_steps):
    key = jax.random.PRNGKey(seed)
    if robot == "quadrotor" and reference == "lissajous":
        paths = (Path("ref_state_list.npy"), Path("ref_u_list.npy"))
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError("Quadrotor Lissajous export needs ref_state_list.npy and "
                                    "ref_u_list.npy in the working directory. Alternatively use "
                                    "--reference random_walk.")
        states, actions = (np.load(path, mmap_mode="r", allow_pickle=False) for path in paths)
        if (states.ndim != 3 or states.shape[2] != 18 or actions.ndim != 3
                or actions.shape[2] != 4 or states.shape[:2] != actions.shape[:2]):
            raise ValueError("Expected matching trajectory banks shaped (N,T,18) and (N,T,4).")
        if trajectory_index >= states.shape[0] or num_steps > states.shape[1]:
            raise ValueError("trajectory-index or num-timesteps exceeds the quadrotor trajectory bank.")
        state, _ = env.reset(key, traj_index=jnp.array([trajectory_index]))
    else:
        state, _ = env.reset(key)
    return state


def reference_values(state):
    # Do not repeat entire quadrotor trajectory banks at every timestep.
    return {field.name: getattr(state, field.name) for field in fields(state)
            if (field.name.startswith("ref_") and not field.name.endswith("_list"))
            or field.name in ("time", "rnd_key")}


def make_reference(env, state, rng, num_steps):
    """Generate one reference tape, independent of the policies being compared.

    Match the generator's actual-state fields to its reference fields. In particular,
    the released Astrobee random-walk reference update reads actual omega; this
    supplies reference omega there without modifying the training environment.
    """
    def advance(carry, unused):
        current, key = carry
        key, step_key = jax.random.split(key)
        matched = {name: getattr(current, "ref_" + name) for name in ("pos", "vel", "rotm", "omega")
                   if hasattr(current, "ref_" + name)}
        follower = current.replace(**matched)
        following, _, _, _, _ = env.step(step_key, follower, jnp.zeros(env.action_space().shape))
        return (following, key), reference_values(following)
    _, tape = jax.lax.scan(advance, (state, rng), None, length=num_steps)
    return tape


def rollout(policy, state, tape, rng):
    def advance(carry, reference):
        current, key = carry
        key, step_key = jax.random.split(key)
        distribution, _ = policy.model.apply({"params": policy.params}, policy.env._get_obs(current))
        action = distribution.mean()
        following, _, _, _, _ = policy.env.step(step_key, current, action)
        following = following.replace(**reference)
        record = {name: getattr(following, name) for name in ("pos", "vel", "rotm", "omega")
                  if hasattr(following, name)}
        record.update(reference)
        record.update(actions=action, bound_violation=policy.env._is_terminal_error(following))
        return (following, key), record
    _, result = jax.lax.scan(advance, (state, rng), tape)
    return jax.tree_util.tree_map(np.asarray, result)


def finite_rollouts(rollouts):
    """Keep a shared finite prefix if a controller becomes numerically unstable."""
    failures = {}
    for label, data in rollouts.items():
        valid = np.logical_and.reduce([np.isfinite(value).reshape(len(value), -1).all(axis=1)
                                       for value in data.values()])
        bad = np.flatnonzero(~valid)
        failures[label] = int(bad[0]) + 1 if len(bad) else None
    length = min(next(iter(rollouts.values()))["pos"].shape[0],
                 min((step - 1 for step in failures.values() if step is not None), default=float("inf")))
    if length == 0:
        raise ValueError("A policy produced nonfinite data on the first step; no comparable rollout to export.")
    return {label: {key: value[:length] for key, value in data.items()}
            for label, data in rollouts.items()}, failures


def tracking_metrics(label, data):
    distance = np.linalg.norm(data["pos"].astype(np.float64) - data["ref_pos"], axis=1)
    velocity_error = np.linalg.norm(data["vel"].astype(np.float64) - data["ref_vel"], axis=1)
    violations = np.flatnonzero(data["bound_violation"])
    result = dict(mode=label, position_rmse_m=float(np.sqrt(np.mean(distance ** 2))),
                  position_max_m=float(distance.max()),
                  velocity_rmse_mps=float(np.sqrt(np.mean(velocity_error ** 2))),
                  first_bound_violation_step=int(violations[0]) + 1 if len(violations) else None)
    result["attitude_rmse_deg"] = None
    if "rotm" in data:
        relative = np.swapaxes(data["ref_rotm"].astype(np.float64), 1, 2) @ data["rotm"]
        angle = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
        result["attitude_rmse_deg"] = float(np.rad2deg(np.sqrt(np.mean(angle ** 2))))
    return result


def export_rollouts(output_dir, policies, rollouts, pair, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = next(iter(rollouts.values()))
    np.save(output_dir / "ref_pos.npy", reference["ref_pos"])
    if "ref_rotm" in reference:
        np.save(output_dir / "ref_quat.npy", Rotation.from_matrix(reference["ref_rotm"]).as_quat())
    for label, data in rollouts.items():
        np.save(output_dir / f"{label}_pos.npy", data["pos"])
        if "rotm" in data:
            np.save(output_dir / f"{label}_quat.npy", Rotation.from_matrix(data["rotm"]).as_quat())
        np.savez_compressed(output_dir / f"{label}_rollout.npz", **data)
    if pair:
        for slot, label in zip(("baseline", "equivariant"), pair):
            data = rollouts[label]
            np.save(output_dir / f"{slot}_pos.npy", data["pos"])
            if "rotm" in data:
                np.save(output_dir / f"{slot}_quat.npy", Rotation.from_matrix(data["rotm"]).as_quat())
        if metadata["robot"] == "particle":
            for filename, data in (("anim_ref", reference["ref_pos"]),
                                   ("anim_baseline", rollouts[pair[0]]["pos"]),
                                   ("anim_eq", rollouts[pair[1]]["pos"])):
                np.save(output_dir / f"{filename}.npy", data)
    metrics = [tracking_metrics(label, data) for label, data in rollouts.items()]
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    metadata.update(rviz_pair=pair, quaternion_order="xyzw", metrics=metrics, policies={
        policy.label: dict(checkpoint=str(policy.checkpoint), mode=policy.config["EQUIVARIANT"],
                           observation_inputs=policy.env.observation_space().shape[0],
                           training_config=policy.config) for policy in policies})
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metrics


def main(argv=None):
    args = parse_args(argv)
    requests = checkpoint_requests(args.env, args.checkpoint)
    pair = args.rviz_pair or ([next(iter(requests)), next(reversed(requests))] if len(requests) > 1 else None)
    if pair and any(label not in requests for label in pair):
        raise ValueError("Both --rviz-pair modes must have a requested checkpoint.")
    policies = load_policies(args.env, args.reference, requests, args.train_seed)
    env = policies[0].env
    state = initial_state(env, args.env, args.reference, args.seed, args.trajectory_index, args.num_timesteps)
    rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1)
    tape = make_reference(env, state, rng, args.num_timesteps)
    rollouts = {}
    for policy in policies:
        print(f"Evaluating {policy.label}: mode {policy.config['EQUIVARIANT']}, "
              f"{policy.env.observation_space().shape[0]} inputs, {policy.checkpoint}", flush=True)
        rollouts[policy.label] = rollout(policy, state, tape, rng)
    rollouts, failures = finite_rollouts(rollouts)
    length = len(next(iter(rollouts.values()))["pos"])
    output_dir = args.output_dir or (REPO_ROOT / "ros2_ws/src/trajectory_visualization/trajectory_visualization/data" / args.env)
    metadata = dict(robot=args.env, reference=args.reference, seed=args.seed, train_seed=args.train_seed,
                    trajectory_index=args.trajectory_index, dt=env.dt, requested_steps=args.num_timesteps,
                    exported_steps=length, first_nonfinite_step=failures, automatic_resets=False,
                    reference_generation="shared reference-matched generator replay",
                    common_task_config_from=policies[0].label, termination_bound=env.termination_bound)
    metrics = export_rollouts(output_dir, policies, rollouts, pair, metadata)
    print("\nMode                Position RMSE (m)   Max distance (m)   Attitude RMSE (deg)")
    for item in metrics:
        angle = item["attitude_rmse_deg"]
        angle_text = f"{angle:.4f}" if angle is not None else "n/a"
        print(f"{item['mode']:<19} {item['position_rmse_m']:>16.4f} {item['position_max_m']:>18.4f} "
              f"{angle_text:>21}")
    print(f"Saved {length} steps ({length * env.dt:.2f} s) to {output_dir.resolve()}")
    if length < args.num_timesteps:
        print(f"All exports truncated to a shared finite prefix; first nonfinite steps: {failures}")
    if pair:
        print(f"RViz slots: baseline={pair[0]}, equivariant={pair[1]}. All requested modes are saved separately.")
    return metrics


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"eval_vis: {error}") from error

