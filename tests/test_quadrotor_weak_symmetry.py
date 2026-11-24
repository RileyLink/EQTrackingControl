"""Check the paper's weak projection on the original quadrotor environments."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from envs.base_envs import SE2xRQuadState, SE2xRQuadRandomWalkState
from envs.quadrotor_envs import (
    SE2xRQuadPosition, SE2xRQuadRandomWalk, SE2xRQuadLissajous,
    strong_so2_observation, weak_se23_observation,
)


ENVIRONMENTS = (SE2xRQuadPosition, SE2xRQuadRandomWalk, SE2xRQuadLissajous)


def example_state(state_type=SE2xRQuadState):
    values = dict(
        time=jnp.array(.2), pos=jnp.array([.4, -.2, 1.2]),
        vel=jnp.array([.3, -.1, .2]),
        rotm=jnp.asarray(Rotation.from_rotvec([.11, -.08, .14]).as_matrix()),
        omega=jnp.array([.1, -.04, .03]),
        ref_pos=jnp.array([.2, .1, 1.]), ref_vel=jnp.array([.1, .2, -.1]),
        ref_rotm=jnp.asarray(Rotation.from_rotvec([.05, .12, -.2]).as_matrix()),
        ref_omega=jnp.array([-.06, .04, .03]),
        action=jnp.full(4, .6573), ref_action=jnp.array([.65, .67, .66, .65]),
    )
    if state_type is SE2xRQuadRandomWalkState:
        values["rnd_key"] = jax.random.PRNGKey(7)
    return state_type(**values)


class OriginalQuadrotorWeakSymmetryTests(unittest.TestCase):
    def test_weak_observation_matches_paper_group_quotient(self):
        state = example_state()
        # The paper's extended pose has world velocity R nu in column 3.
        def extended_pose(rotation, position, velocity):
            matrix = np.eye(5)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = velocity
            matrix[:3, 4] = position
            return matrix

        actual = extended_pose(state.rotm, state.pos, state.vel)
        desired = extended_pose(state.ref_rotm, state.ref_pos, state.ref_vel)
        error = np.linalg.solve(desired, actual)
        angular_error = state.omega - error[:3, :3].T @ state.ref_omega
        expected = np.concatenate([
            state.omega, angular_error, error[:3, :3].ravel(),
            error[:3, 4], error[:3, 3], state.ref_action,
        ])
        np.testing.assert_allclose(weak_se23_observation(state), expected, atol=2e-7)

    def test_modes_preserve_original_full_and_strong_features(self):
        state = example_state()
        full = np.concatenate([
            state.pos, state.vel, state.rotm.ravel(), state.omega,
            state.ref_pos, state.ref_vel, state.ref_rotm.ravel(),
            state.ref_omega, state.ref_action,
        ])
        for env_type in ENVIRONMENTS:
            with self.subTest(environment=env_type.__name__):
                np.testing.assert_array_equal(env_type(equivariant=0)._get_obs(state), full)
                frame = state.ref_rotm if env_type is SE2xRQuadPosition else state.rotm
                other = state.rotm if env_type is SE2xRQuadPosition else state.ref_rotm
                released = jnp.hstack([
                    frame.T @ (state.ref_pos - state.pos), state.ref_vel, state.vel,
                    (frame.T @ other).ravel(), state.ref_omega, state.omega,
                    state.ref_action, state.rotm[:, 2],
                ])
                np.testing.assert_array_equal(env_type(equivariant=1)._get_obs(state), released)

    def test_declared_shapes_and_jitted_observations(self):
        state = example_state()
        for env_type in ENVIRONMENTS:
            for mode, size in [(0, 40), (1, 31), (2, 25), (3, 31)]:
                with self.subTest(environment=env_type.__name__, mode=mode):
                    env = env_type(equivariant=mode)
                    obs = jax.jit(env._get_obs)(state)
                    self.assertEqual(obs.shape, (size,))
                    self.assertEqual(env.observation_space().shape, (size,))
                    self.assertTrue(np.isfinite(obs).all())

    def test_weak_observation_is_invariant_under_se23_action(self):
        state = example_state()
        rotation = jnp.asarray(Rotation.from_rotvec([.7, -.4, .5]).as_matrix())
        shift = jnp.array([2., -3., .4])
        boost = jnp.array([1.2, -.6, .3])
        transformed = state.replace(
            pos=rotation @ state.pos + shift, vel=rotation @ state.vel + boost,
            rotm=rotation @ state.rotm,
            ref_pos=rotation @ state.ref_pos + shift,
            ref_vel=rotation @ state.ref_vel + boost,
            ref_rotm=rotation @ state.ref_rotm,
        )
        np.testing.assert_allclose(
            weak_se23_observation(state), weak_se23_observation(transformed), atol=4e-7,
        )

    def test_corrected_strong_is_invariant_under_yaw_and_translation(self):
        state = example_state()
        rotation = jnp.asarray(Rotation.from_euler("z", .8).as_matrix())
        shift = jnp.array([2., -3., .4])
        transformed = state.replace(
            pos=rotation @ state.pos + shift, vel=rotation @ state.vel,
            rotm=rotation @ state.rotm,
            ref_pos=rotation @ state.ref_pos + shift,
            ref_vel=rotation @ state.ref_vel, ref_rotm=rotation @ state.ref_rotm,
        )
        np.testing.assert_allclose(
            strong_so2_observation(state), strong_so2_observation(transformed), atol=4e-7,
        )

    def test_corrected_strong_retains_tilt_relative_to_gravity(self):
        state = example_state()
        rotation = jnp.asarray(Rotation.from_rotvec([.7, -.4, .5]).as_matrix())
        tilted = state.replace(
            pos=rotation @ state.pos, vel=rotation @ state.vel,
            rotm=rotation @ state.rotm, ref_pos=rotation @ state.ref_pos,
            ref_vel=rotation @ state.ref_vel, ref_rotm=rotation @ state.ref_rotm,
        )
        self.assertGreater(float(jnp.linalg.norm(
            strong_so2_observation(state)[-3:] - strong_so2_observation(tilted)[-3:])), .1)
        np.testing.assert_allclose(
            weak_se23_observation(state), weak_se23_observation(tilted), atol=4e-7,
        )

    def test_common_velocity_boost_distinguishes_original_and_weak(self):
        state = example_state()
        boost = jnp.array([1.2, -.6, .3])
        boosted = state.replace(vel=state.vel + boost, ref_vel=state.ref_vel + boost)
        weak = SE2xRQuadRandomWalk(equivariant=2)
        for mode in [1, 3]:
            strong = SE2xRQuadRandomWalk(equivariant=mode)
            self.assertGreater(float(jnp.linalg.norm(strong._get_obs(state) - strong._get_obs(boosted))), 1.)
        np.testing.assert_allclose(weak._get_obs(state), weak._get_obs(boosted), atol=1e-7)

    def test_physical_step_rewards_and_termination_are_identical(self):
        for env_type, state_type in [
            (SE2xRQuadPosition, SE2xRQuadState),
            (SE2xRQuadRandomWalk, SE2xRQuadRandomWalkState),
        ]:
            with self.subTest(environment=env_type.__name__):
                state = example_state(state_type)
                action, key = jnp.array([.66, .65, .67, .65]), jax.random.PRNGKey(11)
                original = env_type(equivariant=1).step(key, state, action)
                for mode in [2, 3]:
                    reduced = env_type(equivariant=mode).step(key, state, action)
                    # Only the observation at index 1 is allowed to differ.
                    for index in [0, 2, 3, 4]:
                        for left, right in zip(jax.tree_util.tree_leaves(original[index]),
                                               jax.tree_util.tree_leaves(reduced[index])):
                            np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
