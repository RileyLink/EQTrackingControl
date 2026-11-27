"""Run, resume, and hand off the eight symmetry policies on a GPU cluster."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

from rl.cluster_jobs import JOBS, job_config


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "flax", "optax",
            "orbax-checkpoint", "numpy", "scipy", "gymnax", "distrax", "chex",
            "nvidia-cuda-nvcc-cu12", "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12")


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def source_info():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    # Upstream tracks egg-info files that editable installation regenerates.
    # They are build metadata, not training source; package versions are recorded separately.
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", ".",
                                     ":(glob,exclude)**/*.egg-info/**"],
                                    cwd=REPO_ROOT, text=True).splitlines()
    return dict(commit=commit, dirty_files=dirty)


def runtime_info(require_gpu=True):
    from rl.preflight import check_device
    device_check = check_device(require_gpu)
    dependencies = {}
    for name in VERSIONS:
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return dict(python=sys.version, platform=platform.platform(), dependencies=dependencies,
                devices=device_check["devices"], device_check=device_check,
                slurm={key: os.environ[key] for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID",
                       "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST") if key in os.environ})


@contextmanager
def training_lock(path):
    """A process lock released by the OS even after a killed/preempted job."""
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError(f"Another process is training in {path.parent}") from error
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f"Another process is training in {path.parent}") from error
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def restore_runner(path, template):
    import jax
    import orbax.checkpoint as ocp
    # Preserve stored dtypes instead of casting returns to a reset template's
    # weak integer zeros. Only placement follows the current evaluation device.
    restore_args = jax.tree_util.tree_map(
        lambda value: ocp.ArrayRestoreArgs(sharding=value.sharding), template,
    )
    return ocp.PyTreeCheckpointer().restore(
        str(path), item=template, restore_args=restore_args,
    )


def stable_runner(runner):
    """Give reset logging accumulators the dtypes produced by the first step."""
    import jax
    import jax.numpy as jnp
    logging_state = runner[1].replace(
        episode_returns=runner[1].episode_returns.astype(jnp.float32),
        returned_episode_returns=runner[1].returned_episode_returns.astype(jnp.float32),
    )
    runner = (runner[0], logging_state, *runner[2:])
    return jax.tree_util.tree_map(lambda value: jnp.asarray(value, dtype=value.dtype), runner)


def checkpoint_path(run_dir, relative):
    candidate = run_dir / relative
    resolved = candidate.resolve()
    allowed = (relative == "model_final" or
               (Path(relative).parent == Path("resume") and Path(relative).name.startswith("step_")
                and Path(relative).name[5:].isdigit()))
    if not allowed or run_dir.resolve() not in resolved.parents or candidate.is_symlink():
        raise ValueError(f"Invalid managed checkpoint path: {candidate}")
    return resolved


def prune_checkpoints(run_dir, latest):
    resume = run_dir / "resume"
    if not resume.exists():
        return
    candidates = sorted(path for path in resume.iterdir()
                        if path.is_dir() and path.name.startswith("step_") and path.name[5:].isdigit())
    keep = set(candidates[-2:])
    for candidate in candidates:
        if candidate not in keep and candidate != latest:
            # Resolve and constrain every recursive deletion before removing our old state.
            checked = checkpoint_path(run_dir, f"resume/{candidate.name}")
            shutil.rmtree(checked)


def save_curves(path, curves, batch):
    import numpy as np
    arrays = {key: np.concatenate(value, axis=1) for key, value in curves.items()}
    count = arrays["return_mean"].shape[1]
    arrays.update(mean_rewards=arrays["return_mean"].mean(axis=0),
                  std_rewards=arrays["return_mean"].std(axis=0),
                  mean_terminal_timesteps=arrays["length_mean"].mean(axis=0),
                  x_vals_rew=np.arange(1, count + 1) * batch,
                  x_vals_ts=np.arange(1, count + 1) * batch)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def finalize_run(run_dir, job, config, metadata, progress, runner):
    import jax
    import numpy as np
    for value in jax.tree_util.tree_leaves(runner[0].params):
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError("Final policy parameters are nonfinite; this run cannot be marked complete.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with np.load(run_dir / "training_data.npz") as curves:
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(curves["x_vals_rew"], curves["mean_rewards"])
        axis.set(xlabel="Environment steps", ylabel="Mean held episode return", title=config["EXP_NAME"])
        axis.grid(True)
        figure.tight_layout()
        figure.savefig(run_dir / "training_curve.png", dpi=150)
        plt.close(figure)
    progress.update(completed=True, finished_at=now())
    atomic_json(run_dir / "progress.json", progress)
    names = ["config.txt", "run_metadata.json", "progress.json", "training_data.npz", "training_curve.png"]
    names += [path.relative_to(run_dir).as_posix() for path in (run_dir / "model_final").rglob("*")
              if path.is_file()]
    files = {name: dict(sha256=file_digest(run_dir / name), size=(run_dir / name).stat().st_size)
             for name in sorted(names)}
    atomic_json(run_dir / "SUCCESS.json", dict(task=job.index, robot=job.robot, label=job.label,
                seed=config["seed"], smoke=config["SMOKE_TEST"], config_sha256=digest_config(config),
                source_commit=metadata["source"]["commit"], source_dirty=bool(metadata["source"]["dirty_files"]),
                completed_updates=progress["completed_updates"],
                actual_timesteps=progress["completed_updates"] * config["NUM_ENVS"] * config["NUM_STEPS"],
                observation_inputs=job.inputs, files=files))


def train_job(args):
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    from rl.train_policy import make_train

    config = job_config(args.task_index, args.seed, args.smoke)
    job = JOBS[args.task_index]
    source = source_info()
    if source["dirty_files"] and not args.allow_dirty:
        raise ValueError("Working tree is dirty. Commit changes before production training; "
                         "--allow-dirty is for local validation only.")
    if args.allow_dirty and not args.smoke:
        raise ValueError("--allow-dirty is restricted to smoke tests.")
    environment = runtime_info(require_gpu=not args.allow_cpu)
    from rl.preflight import check_rewards
    effective_reward = check_rewards(config)
    run_dir = args.output_root.expanduser().resolve() / job.name(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    with training_lock(run_dir / ".training.lock"):
        config_path = run_dir / "config.txt"
        metadata_path = run_dir / "run_metadata.json"
        fingerprint = digest_config(config)
        if config_path.exists():
            import ast
            if digest_config(ast.literal_eval(config_path.read_text())) != fingerprint:
                raise ValueError("Existing run has a different configuration. Use a new output root.")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata["configuration_sha256"] != fingerprint:
                raise ValueError("Existing run metadata has a different configuration.")
            if metadata["source"]["commit"] != source["commit"]:
                raise ValueError("Resume requires the original source commit recorded in run_metadata.json.")
        else:
            if (run_dir / "progress.json").exists() or (run_dir / "model_final").exists() or (run_dir / "resume").exists():
                raise ValueError("Existing checkpoint data has lost its source metadata.")
            metadata = dict(created_at=now(), task=job.index, source=source, runtime=environment,
                            effective_reward=effective_reward,
                            configuration_sha256=fingerprint, observation_inputs=job.inputs,
                            training_style="from_scratch_then_exact_state_resume",
                            metrics="five scalars per PPO update; no policy-loss changes")
            atomic_json(metadata_path, metadata)
        if not config_path.exists():
            temporary_config = config_path.with_suffix(".tmp")
            temporary_config.write_text(repr(config) + "\n", encoding="utf-8")
            os.replace(temporary_config, config_path)
        progress_file = run_dir / "progress.json"
        progress = json.loads(progress_file.read_text()) if progress_file.exists() else dict(
            completed=False, completed_updates=0, latest_checkpoint=None, training_seconds=0.,
            configuration_sha256=fingerprint)
        if progress["configuration_sha256"] != fingerprint:
            raise ValueError("Progress/configuration fingerprint mismatch.")
        if progress.get("completed") and (run_dir / "SUCCESS.json").is_file():
            print(f"Already complete: {run_dir}", flush=True)
            return 0
        keys = jax.random.split(jax.random.PRNGKey(args.seed), config["num_seeds"])
        initial = make_train({**config, "INITIALIZE_ONLY": True})
        runner = stable_runner(jax.block_until_ready(jax.jit(jax.vmap(initial))(keys))["runner_state"])
        if runner[2].shape[-1] != job.inputs:
            raise ValueError("Environment observation size differs from the campaign specification.")
        done = int(progress["completed_updates"])
        if done:
            runner = restore_runner(checkpoint_path(run_dir, progress["latest_checkpoint"]), runner)
            if not np.all(np.asarray(runner[3]) == done):
                raise ValueError("Checkpoint iteration does not match progress.json.")
        total = config["NUM_UPDATES"]
        if done < 0 or done > total:
            raise ValueError("Invalid completed-update count.")
        curve_names = ("return_mean", "completed_return_mean", "length_mean", "length_std", "episode_count")
        curves = {name: [] for name in curve_names}
        if done:
            with np.load(run_dir / "training_data.npz") as prior:
                for name in curve_names:
                    if prior[name].shape[1] < done:
                        raise ValueError("Training curves are shorter than the committed checkpoint.")
                    curves[name].append(prior[name][:, :done].copy())
        stopping = []
        def request_stop(signum, frame):
            stopping.append(signum)
        handlers = {}
        for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
            if hasattr(signal, name):
                number = getattr(signal, name)
                handlers[number] = signal.signal(number, request_stop)
        compiled = {}
        chunks = 0
        batch = config["NUM_ENVS"] * config["NUM_STEPS"]
        try:
            while done < total:
                count = min(args.updates_per_chunk, total - done)
                if count not in compiled:
                    print(f"Compiling task {job.index}, {count} updates per call...", flush=True)
                    function = jax.jit(jax.vmap(make_train({**config, "UPDATES_PER_CALL": count})))
                    compiled[count] = function.lower(keys, runner).compile()
                start = time.perf_counter()
                output = jax.block_until_ready(compiled[count](keys, runner))
                progress["training_seconds"] += time.perf_counter() - start
                runner = output["runner_state"]
                for name in curve_names:
                    curves[name].append(np.asarray(output["metrics"][name]))
                del output
                completed = done + count
                relative = "model_final" if completed == total else f"resume/step_{completed:09d}"
                destination = checkpoint_path(run_dir, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    # A previous crash can leave an uncommitted checkpoint at this next step.
                    # The committed previous state remains intact while this orphan is replaced.
                    if progress["latest_checkpoint"] == relative:
                        raise ValueError("Refusing to overwrite the committed resume checkpoint.")
                    shutil.rmtree(checkpoint_path(run_dir, relative))
                ocp.PyTreeCheckpointer().save(str(destination), runner)
                save_curves(run_dir / "training_data.npz", curves, batch)
                progress.update(completed_updates=completed, latest_checkpoint=relative, updated_at=now())
                atomic_json(progress_file, progress)
                prune_checkpoints(run_dir, destination)
                done = completed
                chunks += 1
                print(f"{job.name(args.seed)}: {done * batch:,}/{total * batch:,} steps; "
                      f"saved {relative}; training seconds={progress['training_seconds']:.1f}", flush=True)
                if stopping or (args.stop_after_chunks and chunks >= args.stop_after_chunks):
                    break
            if done == total:
                finalize_run(run_dir, job, config, metadata, progress, runner)
                print(f"COMPLETE: {run_dir}", flush=True)
                return 0
            print(f"Checkpointed pause at {done * batch:,} steps; rerun the same command to resume.", flush=True)
            return 75 if stopping else 0
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)


def plan(args):
    print("Campaign v2: original SLURM presets, attitude weight wired, separate fresh run roots.")
    print("Task  Robot       Label              Mode  Inputs  Layers  Units  Requested steps")
    for job in JOBS:
        config = job_config(job.index, args.seed, args.smoke)
        print(f"{job.index:<5} {job.robot:<11} {job.label:<18} {job.mode:<5} {job.inputs:<7} "
              f"{config['NUM_LAYERS']:<7} {config['NUM_NODES']:<6} {config['TOTAL_TIMESTEPS']:,}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="Print all eight configurations without importing JAX.")
    planning.add_argument("--seed", type=int, default=0)
    planning.add_argument("--smoke", action="store_true")
    check = commands.add_parser("check", help="Check the pinned environment and visible GPU.")
    check.add_argument("--allow-cpu", action="store_true")
    training = commands.add_parser("train", help="Train one array task; resume automatically if interrupted.")
    training.add_argument("--task-index", type=int, choices=range(8), required=True)
    training.add_argument("--output-root", type=Path, required=True)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--smoke", action="store_true")
    training.add_argument("--allow-cpu", action="store_true")
    training.add_argument("--allow-dirty", action="store_true")
    training.add_argument("--updates-per-chunk", type=int, default=128)
    training.add_argument("--stop-after-chunks", type=int, default=0)
    package = commands.add_parser("package", help="Collect all eight complete checkpoint directories.")
    package.add_argument("--output-root", type=Path, required=True)
    package.add_argument("--archive", type=Path, required=True)
    package.add_argument("--seed", type=int, default=0)
    package.add_argument("--allow-smoke", action="store_true")
    status = commands.add_parser("status", help="Inspect all eight runs without needing a GPU.")
    status.add_argument("--output-root", type=Path, required=True)
    status.add_argument("--seed", type=int, default=0)
    status.add_argument("--verify", action="store_true", help="Require and checksum-verify all eight completed runs.")
    receive = commands.add_parser("receive", help="Verify and unpack a returned archive.")
    receive.add_argument("--archive", type=Path, required=True)
    receive.add_argument("--destination", type=Path, required=True)
    receive.add_argument("--sha256", help="Expected archive SHA-256; otherwise read its .sha256 sidecar.")
    export = commands.add_parser("export", help="Evaluate one robot's four returned policies.")
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--env", choices=("quadrotor", "astrobee"), required=True)
    export.add_argument("--seed", type=int, default=0, help="Training-seed suffix of the returned folders.")
    export.add_argument("--eval-seed", type=int, default=0)
    export.add_argument("--num-timesteps", type=int, default=2000)
    export.add_argument("--output-dir", type=Path)
    export.add_argument("--rviz-pair", nargs=2, default=["corrected_strong", "weak"])
    args = parser.parse_args(argv)
    if getattr(args, "seed", 0) < 0 or getattr(args, "updates_per_chunk", 1) <= 0 or getattr(args, "stop_after_chunks", 0) < 0:
        parser.error("seed/stop-after-chunks must be nonnegative; updates-per-chunk must be positive")
    if args.command == "plan":
        plan(args)
        return 0
    if args.command == "check":
        from contextlib import redirect_stdout
        import io
        from rl.preflight import check_rewards
        result = runtime_info(not args.allow_cpu)
        with redirect_stdout(io.StringIO()):
            result["task_checks"] = {job.name(0): check_rewards(job_config(job.index)) for job in JOBS}
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "train":
        return train_job(args)
    from rl import cluster_artifacts
    if args.command == "status":
        return cluster_artifacts.status_runs(args.output_root, args.seed, args.verify)
    if args.command == "package":
        cluster_artifacts.package_runs(args.output_root, args.archive, args.seed, args.allow_smoke)
    elif args.command == "receive":
        cluster_artifacts.receive_archive(args.archive, args.destination, args.sha256)
    elif args.command == "export":
        from rl.eval_vis import main as evaluate
        arguments = ["--env", args.env, "--reference", "random_walk", "--seed", str(args.eval_seed),
                     "--num-timesteps", str(args.num_timesteps), "--rviz-pair", *args.rviz_pair]
        for job in JOBS:
            if job.robot == args.env:
                arguments += ["--checkpoint", f"{job.label}={args.output_root / job.name(args.seed)}"]
        if args.output_dir:
            arguments += ["--output-dir", str(args.output_dir)]
        evaluate(arguments)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"cluster_train: {error}") from error
