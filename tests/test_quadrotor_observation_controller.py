"""Checks for the geometric controller and gain identification in observation coordinates."""

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_double_thrust import desired_thrust_jet
from envs.quadrotor_smooth import (
    ALLOCATION, State, reference_at, rotation_exp, sample_trajectory, se23_tracking_features,
)
from experiments.policy_structure.lee import DEFAULT_GAINS, lee_terms, lee_wrench
from experiments.policy_structure.observation_lee import (
    command_terms, request, unpack, world_heading_in_desired_frame,
)
from experiments.policy_structure.observation_sparse import FEATURES, allowed_columns, sparse_path, vector_features
from experiments.policy_structure.variable_gains import (
    CONTEXT, fit_actions, predict, state_tangent,
)


def observation(state, ref):
    return jnp.concatenate((se23_tracking_features(state, ref), desired_thrust_jet(ref),
                            (ALLOCATION @ ref.action)[1:]))


def sample():
    trajectory = sample_trajectory(jax.random.PRNGKey(418))
    ref = reference_at(1.2, trajectory)
    state = State(ref.pos, ref.vel, ref.rot, ref.omega, jnp.array(0), jnp.array(0), ref.action)
    return ref, state


class TestObservationController(unittest.TestCase):
    def test_flatness_wrench_on_the_reference(self):
        ref, state = sample()
        np.testing.assert_allclose(request(observation(state, ref)), ALLOCATION @ ref.action, atol=2e-6)

    def test_original_heading_reconstructs_off_path_lee(self):
        ref, state = sample()
        state = state._replace(pos=state.pos+jnp.array([.2, -.3, .1]), vel=state.vel+jnp.array([.4, -.1, .2]),
                               omega=state.omega+jnp.array([.1, -.2, .3]),
                               rot=state.rot @ rotation_exp(jnp.array([-.2, .3, .1])))
        obs = observation(state, ref)
        expected = lee_wrench(lee_terms(state, ref))
        np.testing.assert_allclose(request(obs, heading_jet=world_heading_in_desired_frame(ref)),
                                   expected, atol=2e-6)
        np.testing.assert_allclose(request(obs)[0], expected[0], atol=2e-7)
        self.assertGreater(float(jnp.linalg.norm(request(obs)[1:] - expected[1:])), 1e-5)

    def test_exact_basis_and_nested_feature_names(self):
        ref, state = sample()
        state = state._replace(pos=state.pos+jnp.array([.3, -.2, .1]), vel=state.vel+jnp.array([-.2, .1, .3]))
        obs = observation(state, ref)
        tracking = unpack(obs)
        kp, kv, kr, kw = DEFAULT_GAINS
        terms = command_terms(tracking, kp, kv)
        full = vector_features(tracking, terms, "broad")
        self.assertEqual(len(full), len(FEATURES["broad"]))
        self.assertEqual(len(set(FEATURES["broad"])), len(full))
        for library in FEATURES:
            expected = vector_features(tracking, terms, library)
            np.testing.assert_allclose(full[:len(expected)], expected)
        exact = vector_features(tracking, terms, "exact")
        predicted = [exact[allowed_columns("exact", axis)] @
                     jnp.array([1., -kp, -kv] if axis == 0 else [-kr, -kw, 1., 1., 1.])
                     for axis in range(4)]
        np.testing.assert_allclose(predicted, request(obs), atol=2e-7)

    def test_tangent_holds_world_velocity_and_matches_error_encoding(self):
        ref, state = sample()
        state = state._replace(vel=state.vel+jnp.array([.3, -.2, .1]))
        delta = jnp.array([.001, -.001, .001, .02, .04, -.03])
        changed = state._replace(rot=state.rot@rotation_exp(delta[:3]), omega=state.omega+delta[3:])
        np.testing.assert_allclose(state_tangent(observation(state, ref), delta), observation(changed, ref), atol=2e-7)

    def test_sparse_fit_and_state_dependent_gain_recover_synthetic_support(self):
        rng = np.random.default_rng(102)
        context = np.column_stack((np.ones(1000), rng.normal(size=(1000, len(CONTEXT)-1))))
        errors = rng.uniform(-1., 1., size=(1000, 8))
        ff = rng.normal(size=(1000, 4))
        train, val = np.arange(1000)<700, np.arange(1000)>=700
        coefficients = np.zeros(context.shape[1])
        coefficients[[0, 4, 7]] = [1., -2., 3.]
        y = context @ coefficients
        recovered, _, _ = sparse_path(context[train], y[train], context[val], y[val], 1., .001)
        np.testing.assert_array_equal(recovered != 0, coefficients != 0)
        np.testing.assert_allclose(recovered, coefficients, atol=.001)
        gains = np.zeros((context.shape[1], 8))
        gains[0] = [.6, .8, 3., 4., 5., 1.2, 1.3, 1.4]
        gains[4, 1] = .7
        bias = np.zeros((context.shape[1], 4))
        bias[0, 0] = .04
        target = np.asarray(predict(context, errors, ff, gains, gain_bias=bias))
        (wg, wb), (cg, cb), _ = fit_actions(context, errors, ff, target, train, val, [1., .8, .8, 1.6])
        np.testing.assert_allclose(wg, gains, atol=.001)
        np.testing.assert_allclose(wb, bias, atol=.001)
        self.assertGreater(np.mean((predict(context, errors, ff, cg, gain_bias=cb) - target)**2), .02)


if __name__ == "__main__":
    unittest.main()
