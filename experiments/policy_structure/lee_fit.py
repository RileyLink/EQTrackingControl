"""Experiment 2. Fit scalar and axis-specific gains with the Lee command jet fixed."""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_smooth import at
from experiments.policy_structure.common import (
    Dataset, experiment_arguments, feasible_wrench, take, wrench_controller,
)
from experiments.policy_structure.jacobians import (
    actor_jacobians, sample_indices, save_comparison, state_jacobians, teacher_functions,
)
from experiments.policy_structure.lee import lee_terms
from experiments.policy_structure.predictions import plot_predictions, save_predictions
from experiments.policy_structure.rollouts import run_closed_loop
from experiments.quadrotor_comparison import write_json


def design(state, ref, baseline_gains):
    terms = lee_terms(state, ref, baseline_gains)
    # Axes in the planner's body frame preserve common yaw symmetry.
    body_z_in_ref = ref.rot.T @ terms.body_z
    thrust = jnp.concatenate((-(ref.rot.T @ terms.position) * body_z_in_ref,
                              -(ref.rot.T @ terms.velocity) * body_z_in_ref))
    moment = -jnp.stack((terms.rotation.attitude, terms.rotation.rate), axis=-1)
    offset = jnp.concatenate((terms.thrust_ff[None], terms.rotation.feedforward))
    return thrust, moment, offset


def solve(design_matrix, target):
    coef, _, rank, singular = np.linalg.lstsq(np.asarray(design_matrix, dtype=np.float64),
                                              np.asarray(target, dtype=np.float64), rcond=None)
    condition = (float(singular[0] / singular[-1]) if rank == design_matrix.shape[-1] else None)
    return coef, dict(rank=int(rank), columns=design_matrix.shape[-1], condition=condition,
                      singular_values=singular.tolist())


def fit(thrust, moment, offset, target, *, diagonal):
    residual = target - offset
    if diagonal:
        thrust_gains, thrust_info = solve(thrust, residual[:, 0])
        moment_fits = [solve(moment[:, i, :], residual[:, i + 1]) for i in range(3)]
        moment_gains = np.stack([coef for coef, _ in moment_fits])
        moment_info = {axis: info for axis, (_, info) in zip("xyz", moment_fits)}
    else:
        thrust_scalar = thrust.reshape(-1, 2, 3).sum(axis=-1)
        scalar, thrust_info = solve(thrust_scalar, residual[:, 0])
        thrust_gains = np.repeat(scalar, 3)
        moment_gains, moment_info = solve(moment.reshape(-1, 2), residual[:, 1:].reshape(-1))
    return thrust_gains, moment_gains, dict(thrust=thrust_info, moment=moment_info)


def predict(thrust, moment, offset, thrust_gains, moment_gains):
    return offset + jnp.concatenate(((thrust @ thrust_gains)[..., None],
                                     jnp.sum(moment * moment_gains, axis=-1)), axis=-1)


def run(data, output, *, jacobian_samples=48):
    if jacobian_samples < 1:
        raise ValueError("jacobian-samples must be positive")
    output.mkdir(parents=True, exist_ok=False)
    env = data.env

    def features(state):
        return design(state, at(env.bank, state.trajectory, state.step), data.gains)

    all_features = jax.jit(jax.vmap(features))(data.states)
    train = data.sample_split == "train"
    target = data.rows["rl_wrench"][train]
    requests = {"Lee": data.rows["lee_requested"]}
    functions = teacher_functions(data)
    controllers = {"RL": data.policy.controller(env),
                   "Lee": wrench_controller(env, functions["Lee requested"])}
    fitted = {}
    for name, diagonal in (("Lee fit scalar", False), ("Lee fit diagonal", True)):
        thrust, moment, diagnostics = fit(*(np.asarray(x)[train] for x in all_features), target,
                                          diagonal=diagonal)
        thrust_jax, moment_jax = jnp.asarray(thrust), jnp.asarray(moment)
        requests[name] = np.asarray(predict(*all_features, thrust_jax, moment_jax))

        def request(state, thrust_gains=thrust_jax, moment_gains=moment_jax):
            return predict(*features(state), thrust_gains, moment_gains)

        controllers[name] = wrench_controller(env, request)
        functions[name] = lambda state, request=request: feasible_wrench(request(state))
        fitted[name] = dict(
            kp=thrust[:3].tolist() if diagonal else float(thrust[0]),
            kv=thrust[3:].tolist() if diagonal else float(thrust[3]),
            kR=moment[:, 0].tolist() if diagonal else float(moment[0]),
            kOmega=moment[:, 1].tolist() if diagonal else float(moment[1]),
            fit_diagnostics=diagnostics,
        )

    write_json(output / "gains.json", dict(
        lee=data.gains._asdict(),
        gains=fitted, units=dict(kp="N/m", kv="N s/m", kR="N m", kOmega="N m s/rad"),
        thrust_axes="planner body frame", moment_axes="current body frame",
        command_attitude="Nominal Lee construction evaluated at each state with nominal gains",
    ))
    applied = save_predictions(data, output, requests)
    plot_predictions(data, output, applied)

    indices = sample_indices(data, jacobian_samples, seed=107)
    states = take(data.states, indices)
    _, teacher = actor_jacobians(data, states)
    jacobians = {"RL": teacher, **{name: state_jacobians(states, function)
                                 for name, function in functions.items() if name != "RL"}}
    save_comparison(data, output, indices, jacobians, plots=True)
    run_closed_loop(data, output, controllers)
    data.analysis_manifest(output, action_target="applied RL wrench", training_split="train",
                           jacobian_samples_per_group=jacobian_samples, initial_conditions=("nominal", "stressed"))
    print(f"Saved scalar and diagonal Lee fits to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    experiment_arguments(parser)
    parser.add_argument("--jacobian-samples", type=int, default=48, help="Per split, family, and source")
    args = parser.parse_args()
    run(Dataset.load(args.dataset, args.checkpoint), args.output, jacobian_samples=args.jacobian_samples)


if __name__ == "__main__":
    main()
