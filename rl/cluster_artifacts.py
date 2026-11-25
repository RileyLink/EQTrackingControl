"""Validate and transfer all checkpoint data, including Orbax tensor chunks."""

import ast
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

from rl.cluster_jobs import JOBS, job_config
from rl.cluster_train import digest_config, file_digest


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
            or name != path.as_posix()):
        raise ValueError(f"Unsafe artifact filename: {name!r}")
    return path


def verify_run(output_root, job, seed, allow_smoke=False):
    run_dir = output_root / job.name(seed)
    if run_dir.is_symlink():
        raise ValueError(f"Checkpoint directory must not be a symlink: {run_dir}")
    success_file = run_dir / "SUCCESS.json"
    if not success_file.is_file():
        raise ValueError(f"Task {job.index} is incomplete: {success_file} is missing.")
    success = json.loads(success_file.read_text())
    smoke = success.get("smoke", False)
    if smoke and not allow_smoke:
        raise ValueError("Smoke-test weights cannot be packaged as production policies. "
                         "Use --allow-smoke only for testing the handoff.")
    expected = job_config(job.index, seed, smoke)
    actual = ast.literal_eval((run_dir / "config.txt").read_text())
    if digest_config(expected) != digest_config(actual) or success["config_sha256"] != digest_config(actual):
        raise ValueError(f"Task {job.index} configuration differs from the campaign specification.")
    if success["task"] != job.index or success["observation_inputs"] != job.inputs:
        raise ValueError(f"Task {job.index} identity or observation dimensions do not match.")
    if success["completed_updates"] != expected["NUM_UPDATES"]:
        raise ValueError(f"Task {job.index} has not completed its full update budget.")
    if success.get("source_dirty") and not smoke:
        raise ValueError("Production checkpoints must come from a clean source checkout.")
    names = list(success["files"])
    if not any(name.startswith("model_final/") and "/d/" in name for name in names):
        raise ValueError(f"Task {job.index} is missing Orbax tensor data; metadata alone is insufficient.")
    for name, recorded in success["files"].items():
        safe_name(name)
        path = run_dir / name
        if (run_dir.resolve() not in path.resolve().parents or path.is_symlink()
                or not path.is_file() or path.stat().st_size != recorded["size"]
                or file_digest(path) != recorded["sha256"]):
            raise ValueError(f"Missing or changed checkpoint artifact: {path}")
    progress = json.loads((run_dir / "progress.json").read_text())
    if not progress.get("completed") or progress["latest_checkpoint"] != "model_final":
        raise ValueError(f"Task {job.index} has no completed final checkpoint.")
    return run_dir, success


def package_runs(output_root, archive, seed=0, allow_smoke=False):
    output_root, archive = output_root.expanduser().resolve(), archive.expanduser().resolve()
    if archive.exists() or archive.with_name(archive.name + ".sha256").exists():
        raise ValueError("Archive or checksum already exists. Choose a new archive filename.")
    verified = [verify_run(output_root, job, seed, allow_smoke) for job in JOBS]
    if len({success["source_commit"] for _, success in verified}) != 1:
        raise ValueError("The eight runs were produced by different source commits.")
    if len({success["smoke"] for _, success in verified}) != 1:
        raise ValueError("Cannot mix smoke-test and production policies.")
    files = {}
    for run_dir, success in verified:
        for relative in [*success["files"], "SUCCESS.json"]:
            name = f"{run_dir.name}/{relative}"
            path = run_dir / relative
            files[name] = dict(size=path.stat().st_size, sha256=file_digest(path))
    manifest = dict(format_version=1, seed=seed, smoke=verified[0][1]["smoke"],
                    source_commit=verified[0][1]["source_commit"], tasks=[job.index for job in JOBS],
                    files=files)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".tmp")
    try:
        with tarfile.open(temporary, "w:gz") as bundle:
            payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            entry = tarfile.TarInfo("CAMPAIGN.json")
            entry.size, entry.mode = len(payload), 0o644
            bundle.addfile(entry, io.BytesIO(payload))
            for name in sorted(files):
                path = output_root / name
                entry = tarfile.TarInfo(name)
                entry.size, entry.mode = path.stat().st_size, 0o644
                with path.open("rb") as stream:
                    bundle.addfile(entry, stream)
        os.replace(temporary, archive)
    finally:
        if temporary.exists():
            temporary.unlink()
    checksum = file_digest(archive)
    sidecar = archive.with_name(archive.name + ".sha256")
    sidecar.write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    print(f"Return BOTH files:\n{archive}\n{sidecar}\nSHA-256: {checksum}")
    return manifest


def status_runs(output_root, seed=0, verify=False):
    complete = 0
    for job in JOBS:
        run_dir = output_root / job.name(seed)
        progress_file = run_dir / "progress.json"
        if not progress_file.exists():
            print(f"{job.index}: {job.name(seed)}: no checkpoint yet")
            continue
        progress = json.loads(progress_file.read_text())
        finished = progress.get("completed") and (run_dir / "SUCCESS.json").is_file()
        if finished and verify:
            verify_run(output_root, job, seed, allow_smoke=True)
        complete += int(bool(finished))
        print(f"{job.index}: {job.name(seed)}: {'complete' if finished else 'partial/checkpointed'}; "
              f"{progress['completed_updates']} PPO updates")
    print(f"{complete}/8 complete")
    return 0 if complete == 8 else 1


def receive_archive(archive, destination, expected_checksum=None):
    archive, destination = archive.expanduser().resolve(), destination.expanduser().resolve()
    if destination.exists():
        raise ValueError("Destination already exists. Choose a new directory to preserve existing policies.")
    if expected_checksum is None:
        sidecar = archive.with_name(archive.name + ".sha256")
        if not sidecar.is_file():
            raise ValueError("Provide the archive's .sha256 file alongside it, or pass --sha256 HASH.")
        tokens = sidecar.read_text().split()
        if not tokens:
            raise ValueError("The archive checksum file is empty.")
        expected_checksum = tokens[0]
    if (len(expected_checksum) != 64 or any(c not in "0123456789abcdef" for c in expected_checksum.lower())
            or file_digest(archive) != expected_checksum.lower()):
        raise ValueError("Archive SHA-256 verification failed.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".symmetry-receive-", dir=destination.parent) as staging:
        root = Path(staging) / "runs"
        root.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            names = [member.name for member in members]
            if len(set(names)) != len(names) or "CAMPAIGN.json" not in names:
                raise ValueError("Duplicate archive filenames or missing campaign manifest.")
            if any(not member.isfile() for member in members):
                raise ValueError("Archive must contain regular files only, without links or special entries.")
            for name in names:
                safe_name(name)
            manifest = json.load(bundle.extractfile("CAMPAIGN.json"))
            if manifest.get("format_version") != 1 or manifest.get("tasks") != list(range(8)):
                raise ValueError("Unsupported or incomplete campaign archive.")
            if set(names) != set(manifest["files"]) | {"CAMPAIGN.json"}:
                raise ValueError("Archive members do not match its manifest.")
            for member in members:
                if member.name == "CAMPAIGN.json":
                    continue
                expected = manifest["files"][member.name]
                if member.size != expected["size"]:
                    raise ValueError(f"Archive file size mismatch: {member.name}")
                target = root / member.name
                if root.resolve() not in target.resolve().parents:
                    raise ValueError("Archive member escapes the destination.")
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("xb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                if file_digest(target) != expected["sha256"]:
                    raise ValueError(f"Archive file checksum mismatch: {member.name}")
        verified = [verify_run(root, job, manifest["seed"], allow_smoke=True) for job in JOBS]
        if any(success["source_commit"] != manifest["source_commit"] for _, success in verified):
            raise ValueError("Campaign source commit differs from a policy's source commit.")
        (root / "CAMPAIGN.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if destination.exists():
            raise ValueError("Destination appeared during verification; refusing to replace it.")
        # Both paths are resolved beneath this explicitly selected destination parent.
        if root.resolve().parent.parent != destination.parent.resolve():
            raise ValueError("Invalid staging directory.")
        os.rename(root.resolve(), destination)
    print(f"Verified and received all eight policies in {destination}")
    return manifest
