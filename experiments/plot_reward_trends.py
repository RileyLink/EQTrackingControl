"""Plot longer training rewards with explicitly labelled smoothing and seed bands."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.quadrotor_comparison import COLORS, LABELS, read_csv, write_csv, write_json


def plot(output, window=16):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Wait for all policies to finish before comparing the final trends")
    if window < 1:
        raise ValueError("Smoothing window must be positive")
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    summaries, diagnostics = [], []
    for variant in manifest["variants"]:
        rewards = []
        for seed in manifest["seeds"]:
            rows = read_csv(output / variant / f"seed_{seed}" / "training.csv")
            steps = np.array([int(row["steps"]) for row in rows])
            values = np.array([float(row["step_reward"]) for row in rows])
            if len(values) < window or steps[-1] != manifest["config"]["total_steps"]:
                raise ValueError("Incomplete reward history")
            if rewards:
                np.testing.assert_array_equal(steps, shared_steps)
            shared_steps = steps
            numeric = np.array([[float(value) for value in row.values()] for row in rows])
            if not np.isfinite(numeric).all():
                raise ValueError(f"Non-finite metrics in {variant}, seed {seed}")
            diagnostics.append(dict(variant=variant, seed=seed, updates=len(rows),
                metrics_finite=True, max_approx_kl=max(float(row["approx_kl"]) for row in rows),
                min_reward=float(values.min()), max_reward=float(values.max())))
            rewards.append(values)
        rewards = np.stack(rewards)
        smoothed = np.stack([np.convolve(values, np.ones(window) / window, mode="valid")
                             for values in rewards])
        x = shared_steps[window - 1:] / 1e6
        mean, std = smoothed.mean(axis=0), smoothed.std(axis=0)
        style = {"G_symmetry_thrust_rates": "--", "H_dynamic_thrust_extension": "-.",
                 "I_full_double_thrust_extension": ":", "J_symmetry_double_thrust_extension": "--",
                 "K_full_desired_thrust_extension": (0, (5, 1, 1, 1)),
                 "L_symmetry_desired_thrust_extension": (0, (3, 1, 1, 1))}.get(variant, "-")
        ax.plot(x, mean, color=COLORS[variant], label=LABELS[variant], linewidth=2, linestyle=style)
        if len(rewards) > 1:
            ax.fill_between(x, mean - std, mean + std, color=COLORS[variant], alpha=.13)
        tail_updates = min(64, rewards.shape[1])
        first = rewards[:, :tail_updates].mean(axis=1)
        last = rewards[:, -tail_updates:].mean(axis=1)
        summaries.append(dict(variant=variant, seeds=len(rewards),
                              averaging_updates=tail_updates,
                              averaging_steps=tail_updates * manifest["config"]["num_envs"] * manifest["config"]["rollout_steps"],
                              first_window_reward=float(first.mean()),
                              last_window_reward=float(last.mean()),
                              last_window_seed_std=float(last.std()),
                              improvement=float((last - first).mean())))
    ax.axvline(.262144, color="#94a3b8", linestyle="--", linewidth=1)
    ax.text(.262144 + .015, ax.get_ylim()[0] + .025, "Previous pilot budget", rotation=90,
            color="#64748b", fontsize=9, va="bottom")
    ax.set(title=f"Longer quadrotor training | {len(manifest['variants'])} policies | {len(manifest['seeds'])} seeds\n"
                 f"{window}-update rolling mean; shading = +/- 1 seed standard deviation",
           xlabel="Environment steps per seed [millions]",
           ylabel="Training reward per environment step (higher is better)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8.5, ncol=2, loc="lower right",
              title="NN dimensions: obs = inputs, act = outputs", title_fontsize=9)
    fig.savefig(output / "training_reward_smoothed.png", dpi=160)
    plt.close(fig)
    write_csv(output / "reward_trends.csv", summaries)
    write_json(output / "training_diagnostics.json", diagnostics)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window", type=int, default=16)
    args = parser.parse_args()
    plot(args.output, args.window)
