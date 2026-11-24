"""Measure command sensitivity and tracking dependence on desired thrust inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs.quadrotor_desired_thrust import direct_wrench_action
from envs.quadrotor_smooth import ALLOCATION, GRAVITY, MASS, SE23_TRACKING_ORDER, State, Reference, make_environment
from experiments.quadrotor_comparison import load_policy, write_csv, write_json
from rl.quadrotor_ppo import command, make_evaluator


POLICIES = {
    "B_prime_se23_wrench": Path("runs/se23_b_prime_2m"),
    "L_se23_desired_thrust_extension": Path("runs/se23_bl_2m"),
}
REFERENCES = Path("runs/path_tracking_se23_bl_2m")
MASKS = {
    "unmodified": (), "thrust_to_hover": (0,), "rate_to_zero": (1,),
    "acceleration_to_zero": (2,), "both_derivatives_to_zero": (1, 2),
}


class MaskedActor:
    def __init__(self, model, neutral):
        self.model = model
        self.neutral = neutral

    def apply(self, arguments, observation):
        params, mask = arguments
        return self.model.apply(params, jnp.where(mask, self.neutral, observation))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pre_action_states(frames, bank):
    """Recover exactly the states at which the unmasked actor chose an action."""
    def before(name):
        return np.concatenate([frames[f"initial_{name}"][None], frames[name][:-1]], axis=0)

    active = frames["active"]
    step, trajectory = np.indices(active.shape)
    rotor = np.concatenate([np.asarray(bank.action[:, 0])[None], frames["rotor_command"][:-1]], axis=0)
    return State(*(x[active] for x in (before("position"), before("velocity"), before("rotation"),
                                        before("omega"), trajectory, step, rotor)))


def summarize(rows):
    result = []
    for variant in POLICIES:
        for case in MASKS:
            selected = [r for r in rows if r["variant"] == variant and r["case"] == case]
            if selected:
                result.append(dict(variant=variant, case=case,
                    trials=sum(r["trials"] for r in selected), completed=sum(r["completed"] for r in selected),
                    rmse_cm=float(np.mean([r["rmse_cm"] for r in selected])),
                    paired_delta_rmse_cm=float(np.mean([r["paired_delta_rmse_cm"] for r in selected])),
                    thrust_rms_change_mN=float(np.sqrt(np.mean([r["thrust_rms_change_mN"]**2 for r in selected]))),
                    torque_rms_change_mNm=float(np.sqrt(np.mean([r["torque_rms_change_mNm"]**2 for r in selected])))))
    return result


def plot(output, rows):
    variant = "L_se23_desired_thrust_extension"
    cases = list(MASKS)[1:]
    labels = [r"$F_d$ to hover", r"$\dot F_d$ to zero", r"$\ddot F_d$ to zero", "Both derivatives to zero"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, layout="constrained")
    axes[0].set_yticks(range(len(cases)), labels)
    axes[0].invert_yaxis()
    for ax, metric, title in zip(axes, ("thrust_rms_change_mN", "torque_rms_change_mNm", "paired_delta_rmse_cm"),
                                ("Thrust change [mN RMS]", "Body torque change [mN·m RMS]", "Change in tracking RMSE [cm]")):
        for y, case in enumerate(cases):
            values = [r[metric] for r in rows if r["variant"] == variant and r["case"] == case]
            ax.scatter(values, [y] * len(values), color="#64748b", s=23, zorder=3)
            ax.scatter(np.mean(values), y, color="#2563eb", marker="|", linewidth=2.5, s=170, zorder=4)
        ax.set_xlabel(title)
        ax.axvline(0, color="#94a3b8", linewidth=1)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Which desired thrust inputs does policy L use?\n"
                 "Same-state command changes and paired rollouts · dots are seeds, ticks are seed means")
    fig.savefig(output / "input_effects.png", dpi=180)
    plt.close(fig)


def run(output):
    reference_manifest = json.loads((REFERENCES / "manifest.json").read_text(encoding="utf-8"))
    archive = REFERENCES / "references.npz"
    if digest(archive) != reference_manifest["reference_bank_sha256"]:
        raise ValueError("Reference archive hash differs from the saved evaluation")
    with np.load(archive) as data:
        bank = Reference(*(jnp.asarray(data["reference_" + name]) for name in Reference._fields))
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path("envs/quadrotor_desired_thrust.py"), Path("envs/quadrotor_smooth.py"),
               Path("rl/quadrotor_ppo.py"), Path("experiments/quadrotor_comparison.py")]
    (output / "source").mkdir()
    for source in sources:
        shutil.copy2(source, output / "source" / source.name)
    seeds = reference_manifest["seeds"]
    reward_profile = reference_manifest["reward_profile"]
    architecture, rows, rollouts, checkpoint_hashes = [], [], [], {}
    for variant, training in POLICIES.items():
        env = make_environment(bank, variant, dt=reference_manifest["dt"], reward_profile=reward_profile)
        first_index = len(env.scale) - env.reference_thrust_size - 3
        neutral = jnp.zeros(env.obs_size).at[first_index].set(MASS * GRAVITY / env.scale[first_index])
        model, _, _ = load_policy(training / variant / f"seed_{seeds[0]}")
        actor = MaskedActor(model, neutral)
        evaluate = make_evaluator(env, actor, seed=reference_manifest["initial_condition_seed"], record_state=True)
        observe = jax.jit(jax.vmap(env.observe))

        @jax.jit
        def applied_wrench(params, obs, mask):
            mean, _, _ = actor.apply((params, mask), obs)
            rotors, _, _ = jax.vmap(direct_wrench_action)(command(mean))
            return rotors @ ALLOCATION.T

        for seed in seeds:
            path = training / variant / f"seed_{seed}"
            _, params, config = load_policy(path)
            expected_order = [*SE23_TRACKING_ORDER, *["F_d", "Fdot_d", "Fddot_d"][:env.reference_thrust_size], "tau_d"]
            if (config["environment_settings"]["observation_order"] != expected_order
                    or config["steps"] != reference_manifest["checkpoint_steps"]
                    or config["reward_profile"] != reward_profile):
                raise ValueError(f"Incompatible saved checkpoint {path}")
            checkpoint_hashes[f"{variant}/seed_{seed}"] = digest(path / "policy.msgpack")
            if seed == seeds[0]:
                counts = {key: sum(x.size for x in jax.tree.leaves(node)) for key, node in params["params"].items()}
                actor_count = sum(value for key, value in counts.items() if key.startswith("actor_") or key == "logstd")
                architecture.append(dict(variant=variant, inputs=env.obs_size, hidden_layers=model.layers,
                    width=model.width, actor_params=actor_count, critic_params=sum(counts.values()) - actor_count,
                    total_params=sum(counts.values())))

            blank = np.zeros(env.obs_size, dtype=bool)
            baseline, frames = jax.device_get(evaluate((params, blank)))
            obs = observe(pre_action_states(frames, bank))
            wrench = np.asarray(applied_wrench(params, obs, blank))
            np.testing.assert_allclose(wrench, frames["rotor_command"][frames["active"]] @ np.asarray(ALLOCATION).T,
                                       atol=2e-6, rtol=0.)
            thrust_offset = (np.asarray(obs[:, first_index]) - float(neutral[first_index])) * float(env.scale[first_index])
            for case, indices in MASKS.items():
                if indices and max(indices) >= env.reference_thrust_size:
                    continue
                mask = blank.copy()
                for index in indices:
                    mask[first_index + index] = True
                if indices:
                    metrics, masked = jax.device_get(evaluate((params, mask)))
                    for field in ("position", "velocity", "rotation", "omega"):
                        np.testing.assert_array_equal(frames[f"initial_{field}"], masked[f"initial_{field}"])
                    change = wrench - np.asarray(applied_wrench(params, obs, mask))
                else:
                    metrics, change = baseline, np.zeros_like(wrench)
                if not np.isfinite(metrics).all() or not np.isfinite(change).all():
                    raise FloatingPointError(f"Nonfinite evaluation for {variant}, seed {seed}, {case}")
                rmse = metrics[:, 0] * 100
                delta = rmse - baseline[:, 0] * 100
                failures = metrics[:, 7].astype(bool)
                gain = (float(np.dot(thrust_offset, change[:, 0]) / np.dot(thrust_offset, thrust_offset))
                        if case == "thrust_to_hover" else None)
                rows.append(dict(variant=variant, seed=seed, case=case, trials=len(rmse), completed=int((~failures).sum()),
                    rmse_cm=float(rmse.mean()), paired_delta_rmse_cm=float(delta.mean()),
                    thrust_rms_change_mN=float(1e3 * np.sqrt(np.mean(change[:, 0]**2))),
                    torque_rms_change_mNm=float(1e3 * np.sqrt(np.mean(np.sum(change[:, 1:]**2, axis=1)))),
                    signed_thrust_gain_N_per_N=gain,
                    desired_thrust_rms_about_hover_mN=float(1e3 * np.sqrt(np.mean(thrust_offset**2))),
                    applied_thrust_rms_about_hover_mN=float(1e3 * np.sqrt(np.mean((wrench[:, 0] - MASS * GRAVITY)**2))),
                    applied_torque_rms_mNm=float(1e3 * np.sqrt(np.mean(np.sum(wrench[:, 1:]**2, axis=1))))))
                rollouts.extend(dict(variant=variant, seed=seed, case=case, trajectory=i, rmse_cm=float(rmse[i]),
                    paired_delta_rmse_cm=float(delta[i]), failed=bool(failures[i]), survival_s=float(metrics[i, 8]))
                    for i in range(len(rmse)))
            print(f"{variant} seed {seed} evaluated", flush=True)
    write_csv(output / "network_sizes.csv", architecture)
    write_csv(output / "per_seed.csv", rows)
    write_csv(output / "per_rollout.csv", rollouts)
    summary = summarize(rows)
    write_csv(output / "summary.csv", summary)
    plot(output, rows)
    write_json(output / "manifest.json", dict(status="complete", seeds=seeds, reference_path=str(archive),
        reference_sha256=digest(archive), checkpoint_sha256=checkpoint_hashes, reward_profile=reward_profile,
        masks=MASKS, thrust_neutral_N=MASS * GRAVITY, derivative_neutral=0,
        source_sha256={p.name: digest(p) for p in sources}))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/se23_thrust_input_diagnostics"))
    run(parser.parse_args().output)
