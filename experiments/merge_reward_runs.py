"""Add independently trained policy curves without modifying existing runs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from experiments.quadrotor_comparison import report, write_json
from experiments.plot_reward_trends import plot


def case_settings(run, manifest, variant):
    """Read older runs' settings from their per-policy configs when necessary."""
    if variant in manifest.get("case_settings", {}):
        return manifest["case_settings"][variant]
    path = Path(run) / variant / f"seed_{manifest['seeds'][0]}" / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    return dict(observation_size=config["observation_size"],
                observation_features=config.get("observation_features", {}),
                feature_definition=config.get("feature_definition", "standard; see observation_features and archived source"),
                action_definition=config.get("action_definition", "four-normalized-rotor-commands"),
                settings=config.get("environment_settings", {}))


def merge(runs, output, window=16):
    runs = [Path(path).resolve() for path in runs]
    manifests = [json.loads((path / "manifest.json").read_text(encoding="utf-8")) for path in runs]
    base = manifests[0]
    variants = []
    for path, manifest in zip(runs, manifests):
        if manifest["status"] != "complete":
            raise ValueError(f"Incomplete run: {path}")
        if manifest.get("reward_profile", "legacy") != base.get("reward_profile", "legacy"):
            raise ValueError("Cannot merge training rewards from different reward profiles")
        for field in ["config", "seeds", "horizon", "dt", "dependencies"]:
            if manifest[field] != base[field]:
                raise ValueError(f"Comparison mismatch in {field}: {path}")
        if manifest["bank_sha256"]["train"] != base["bank_sha256"]["train"]:
            raise ValueError("Train reference banks differ")
        reference = path / "train_references.npz"
        if hashlib.sha256(reference.read_bytes()).hexdigest() != base["bank_sha256"]["train"]:
            raise ValueError("Reference archive hash does not match manifest")
        for variant in manifest["variants"]:
            if variant in variants:
                raise ValueError(f"Duplicate policy: {variant}")
            variants.append(variant)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    combined = dict(base)
    for field in ["progress", "wall_seconds", "created_utc", "completed_utc", "source_sha256"]:
        combined.pop(field, None)
    combined.update(variants=variants, merged_utc=datetime.now(timezone.utc).isoformat(),
                    status="complete", source_runs=[], variant_source_run={}, case_settings={},
                    plot_settings={"smoothing_updates": window, "band": "seed population standard deviation"},
                    notes=base.get("notes", []) + [
                        "Merged saved histories; original source runs remain unchanged.",
                        "G changes only inputs; H/I/J change thrust dynamics and action formulation; K/L extend only the desired state and choose f/tau directly."])
    shutil.copy2(runs[0] / "train_references.npz", output / "train_references.npz")
    for index, (path, manifest) in enumerate(zip(runs, manifests)):
        provenance = output / "provenance" / f"run_{index}"
        provenance.mkdir(parents=True)
        shutil.copy2(path / "manifest.json", provenance / "manifest.json")
        shutil.copytree(path / "source", provenance / "source")
        if (path / "provenance").exists():
            shutil.copytree(path / "provenance", provenance / "provenance")
        combined["source_runs"].append({"path": str(path),
            "manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
            "training_wall_seconds": manifest.get("wall_seconds")})
        for variant in manifest["variants"]:
            shutil.copytree(path / variant, output / variant)
            combined["variant_source_run"][variant] = index
            combined["case_settings"][variant] = case_settings(path, manifest, variant)
    # Keep the exact plotting code for the expanded comparison too.
    (output / "source").mkdir()
    combined["source_sha256"] = {}
    for source in [Path(__file__), Path("experiments/quadrotor_comparison.py"),
                   Path("experiments/plot_reward_trends.py"), Path("envs/quadrotor_smooth.py"),
                   Path("envs/quadrotor_thrust_rate.py"), Path("envs/quadrotor_double_thrust.py"),
                   Path("envs/quadrotor_desired_thrust.py"),
                   Path("requirements-quadrotor-lock.txt")]:
        shutil.copy2(source, output / "source" / source.name)
        combined["source_sha256"][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(output / "manifest.json", combined)
    report(output)
    plot(output, window)
    descriptions = ["Original curves are copied from the saved runs without retraining."]
    if "G_symmetry_thrust_rates" in variants:
        descriptions.append("G adds desired and causal interval-averaged actual total-thrust rate to the invariant base.")
    if "H_dynamic_thrust_extension" in variants:
        descriptions.append("H is a first-order thrust integrator with thrust-rate/body-torque actions and a 1 N/s rate limit.")
    if any(name in variants for name in ["I_full_double_thrust_extension", "J_symmetry_double_thrust_extension"]):
        descriptions.append("I/J use the second-order thrust plant with full/invariant observations, states F and Fdot, "
                            "actions Fddot and body torque, and analytic desired thrust-state/input references.")
    if any(name in variants for name in ["K_full_desired_thrust_extension", "L_symmetry_desired_thrust_extension"]):
        descriptions.append("K/L observe full/invariant rigid-body actual/desired features and desired F/Fdot/Fddot/body torque, "
                            "chooses actual f/tau directly, and introduces no actual thrust integrator.")
    descriptions.append("Dynamic extensions change the actuator dynamics as well as the policy inputs. "
                        "No explicit position acceleration/jerk/snap/yaw derivative channels are supplied to G/H/I/J. "
                        "Per-case definitions and actuator limits are retained in the manifest and source provenance.")
    with (output / "REPORT.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## Expanded comparison\n\n" + " ".join(descriptions)
                     + "\n\n![Expanded training reward](training_reward_smoothed.png)\n")
    quoted_output = "'" + str(output).replace("'", "''") + "'"
    instructions = (
        "# Reproduce or extend this graph\n\n"
        "The raw CSV histories and checkpoints for all policies are copied here. "
        "Original source manifests/code are under provenance/run_*/. Plotting source is under source/.\n\n"
        "From the repository root:\n\n```powershell\n"
        "..\\.venv\\Scripts\\python.exe -m experiments.plot_reward_trends " + quoted_output + " --window " + str(window) + "\n```\n\n"
        "Use experiments.merge_reward_runs with additional complete runs to create a new expanded graph. "
        "It checks training settings, seeds and reference-bank hashes and leaves the original data untouched.\n")
    (output / "REPRODUCE.md").write_text(instructions, encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=16)
    args = parser.parse_args()
    merge(args.runs, args.output, args.window)
