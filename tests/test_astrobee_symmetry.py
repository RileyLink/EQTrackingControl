"""Check Astrobee body-frame strong and relative-velocity weak observations."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from envs.base_envs import (
    SE3FAQuadState, SE3FAQuadRandomWalkState, SE3FAQuadLissajousTrackingState,
)
from envs.astrobee_envs import (
    SE3QuadFullyActuatedPosition, SE3QuadFullyActuatedRandomWalk,
    SE3QuadFullyActuatedLissajous, strong_se3_observation, weak_se23_observation,
)


ENVIRONMENTS = (
    (SE3QuadFullyActuatedPosition, SE3FAQuadState, {0: 36, 1: 18, 2: 21, 6: 24}),
    (SE3QuadFullyActuatedRandomWalk, SE3FAQuadRandomWalkState, {0: 42, 1: 30, 2: 27, 6: 30}),
    (SE3QuadFullyActuatedLissajous, SE3FAQuadLissajousTrackingState, {0: 42, 1: 30, 2: 27, 6: 30}),
)


def example_state(state_type):
    values = dict(
        time=jnp.array(.2), pos=jnp.array([.4, -.2, 1.2]),
        vel=jnp.array([.3, -.1, .2]),
        rotm=jnp.asarray(Rotation.from_rotvec([.11, -.08, .14]).as_matrix()),
        omega=jnp.array([.1, -.04, .03]),
        ref_pos=jnp.array([.2, .1, 1.]), ref_vel=jnp.array([.1, .2, -.1]),
        ref_rotm=jnp.asarray(Rotation.from_rotvec([.05, .12, -.2]).as_matrix()),
        ref_omega=jnp.array([-.06, .04, .03]),
    )
    if state_type is not SE3FAQuadState:
        values.update(rnd_key=jax.random.PRNGKey(7),
                      ref_F=jnp.array([.08, -.03, .02]),
                      ref_tau=jnp.array([.01, .02, -.01]),
                      F=jnp.zeros(3), tau=jnp.zeros(3))
    if state_type is SE3FAQuadLissajousTrackingState:
        values.update(amplitudes=jnp.array([.3, .4, .2]),
                      frequencies=jnp.array([.02, .03, .04]),
                      phases=jnp.array([0., .3, .7]))
    return state_type(**values)


def transform(state, rotation, shift, boost):
    return state.replace(
        pos=rotation @ state.pos + shift, vel=rotation @ state.vel + boost,
        rotm=rotation @ state.rotm,
        ref_pos=rotation @ state.ref_pos + shift,
        ref_vel=rotation @ state.ref_vel + boost,
        ref_rotm=rotation @ state.ref_rotm,
    )


class AstrobeeSymmetryTests(unittest.TestCase):
    def test_weak_features_match_extended_pose_quotient(self):
        for _, state_type, _ in ENVIRONMENTS:
            with self.subTest(state=state_type.__name__):
                state = example_state(state_type)
                def extended_pose(rotation, position, velocity):
                    matrix = np.eye(5)
                    matrix[:3, :3], matrix[:3, 3], matrix[:3, 4] = rotation, velocity, position
                    return matrix
                actual = extended_pose(state.rotm, state.pos, state.vel)
                desired = extended_pose(state.ref_rotm, state.ref_pos, state.ref_vel)
                error = np.linalg.solve(actual, desired)
                parts = [error[:3, 4], -error[:3, 3], error[:3, :3].ravel(),
                         state.omega, state.ref_omega]
                include_wrench = state_type is not SE3FAQuadState
                if include_wrench:
                    parts.extend([state.ref_F, state.ref_tau])
                np.testing.assert_allclose(
                    weak_se23_observation(state, include_reference_wrench=include_wrench),
                    np.concatenate(parts), atol=2e-7,
                )

    def test_strong_and_weak_differ_only_in_velocity_features(self):
        for env_type, state_type, _ in ENVIRONMENTS:
            state = example_state(state_type)
            strong = env_type(equivariant=6, symmetry_type=2)._get_obs(state)
            weak = env_type(equivariant=2, symmetry_type=2)._get_obs(state)
            np.testing.assert_array_equal(strong[:3], weak[:3])
            np.testing.assert_array_equal(strong[9:], weak[6:])
            np.testing.assert_allclose(weak[3:6], strong[3:6] - strong[6:9], atol=1e-7)

    def test_original_modes_are_preserved(self):
        for env_type, state_type, _ in ENVIRONMENTS:
            state = example_state(state_type)
            tracking = state_type is not SE3FAQuadState
            full = [state.pos, state.vel, state.rotm.ravel(), state.omega,
                    state.ref_pos, state.ref_vel, state.ref_rotm.ravel(), state.ref_omega]
            if tracking:
                full.extend([state.ref_F, state.ref_tau])
            for setting in range(4):
                with self.subTest(environment=env_type.__name__, setting=setting):
                    np.testing.assert_array_equal(
                        env_type(equivariant=0, symmetry_type=setting)._get_obs(state),
                        jnp.hstack(full),
                    )
                    if tracking:
                        position = state.ref_pos - state.pos
                        if setting in (2, 3):
                            position = state.rotm.T @ position
                        parts = [position, state.vel, state.ref_vel,
                                 (state.rotm.T @ state.ref_rotm).ravel(),
                                 state.omega, state.ref_omega, state.ref_F, state.ref_tau]
                    else:
                        parts = [state.ref_rotm.T @ (state.pos - state.ref_pos),
                                 state.vel - state.ref_vel,
                                 (state.rotm.T @ state.ref_rotm).ravel(),
                                 state.omega - state.ref_omega]
                    np.testing.assert_array_equal(
                        env_type(equivariant=1, symmetry_type=setting)._get_obs(state),
                        jnp.hstack(parts),
                    )

    def test_observation_sizes_jit_and_reset(self):
        for env_type, state_type, sizes in ENVIRONMENTS:
            state = example_state(state_type)
            for mode, size in sizes.items():
                with self.subTest(environment=env_type.__name__, mode=mode):
                    env = env_type(equivariant=mode, symmetry_type=2)
                    observation = jax.jit(env._get_obs)(state)
                    self.assertEqual(observation.shape, (size,))
                    self.assertEqual(env.observation_space().shape, (size,))
                    self.assertEqual(env.reset(jax.random.PRNGKey(10))[1].shape, (size,))
                    self.assertTrue(np.isfinite(observation).all())

    def test_new_modes_require_body_wrench_direct_product_setting(self):
        for env_type, _, _ in ENVIRONMENTS:
            for mode in (2, 6):
                for setting in (0, 1, 3):
                    with self.assertRaisesRegex(ValueError, "symmetry_type=2"):
                        env_type(equivariant=mode, symmetry_type=setting)

    def test_strong_is_invariant_under_se3_action(self):
        rotation = jnp.asarray(Rotation.from_rotvec([.7, -.4, .5]).as_matrix())
        for env_type, state_type, _ in ENVIRONMENTS:
            state = example_state(state_type)
            transformed = transform(state, rotation, jnp.array([2., -3., .4]), jnp.zeros(3))
            env = env_type(equivariant=6, symmetry_type=2)
            np.testing.assert_allclose(env._get_obs(state), env._get_obs(transformed), atol=1e-6)

    def test_weak_also_removes_common_velocity_boost(self):
        rotation = jnp.asarray(Rotation.from_rotvec([.7, -.4, .5]).as_matrix())
        for env_type, state_type, _ in ENVIRONMENTS:
            state = example_state(state_type)
            transformed = transform(state, rotation, jnp.array([2., -3., .4]),
                                    jnp.array([1.2, -.6, .3]))
            weak = env_type(equivariant=2, symmetry_type=2)
            strong = env_type(equivariant=6, symmetry_type=2)
            np.testing.assert_allclose(weak._get_obs(state), weak._get_obs(transformed), atol=1e-6)
            self.assertGreater(float(jnp.linalg.norm(
                strong._get_obs(state) - strong._get_obs(transformed))), 1.)

    def test_new_modes_distinguish_the_missing_heading_example(self):
        zero, identity = jnp.zeros(3), jnp.eye(3)
        yaw90 = jnp.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        first = example_state(SE3FAQuadRandomWalkState).replace(
            pos=zero, ref_pos=zero, vel=jnp.array([1., 0., 0.]), ref_vel=zero,
            rotm=identity, ref_rotm=identity, omega=zero, ref_omega=zero,
            ref_F=zero, ref_tau=zero,
        )
        second = first.replace(rotm=yaw90, ref_rotm=yaw90)
        original = SE3QuadFullyActuatedRandomWalk(equivariant=1, symmetry_type=2)
        np.testing.assert_array_equal(original._get_obs(first), original._get_obs(second))
        for mode in (2, 6):
            env = SE3QuadFullyActuatedRandomWalk(equivariant=mode, symmetry_type=2)
            self.assertGreater(float(jnp.linalg.norm(env._get_obs(first) - env._get_obs(second))), 1.)

    def test_reset_distribution_is_unchanged(self):
        for env_type, _, _ in ENVIRONMENTS:
            key = jax.random.PRNGKey(11)
            original = env_type(equivariant=1, symmetry_type=2).reset(key)[0]
            for mode in (2, 6):
                reduced = env_type(equivariant=mode, symmetry_type=2).reset(key)[0]
                for left, right in zip(jax.tree_util.tree_leaves(original),
                                       jax.tree_util.tree_leaves(reduced)):
                    np.testing.assert_array_equal(left, right)

    def test_action_dynamics_reward_and_termination_are_unchanged(self):
        for env_type, state_type, _ in ENVIRONMENTS:
            state = example_state(state_type)
            action, key = jnp.array([.1, -.08, .04, .02, -.03, .01]), jax.random.PRNGKey(12)
            original = env_type(equivariant=1, symmetry_type=2).step(key, state, action)
            for mode in (2, 6):
                with self.subTest(environment=env_type.__name__, mode=mode):
                    reduced = env_type(equivariant=mode, symmetry_type=2).step(key, state, action)
                    for index in (0, 2, 3, 4):
                        for left, right in zip(jax.tree_util.tree_leaves(original[index]),
                                               jax.tree_util.tree_leaves(reduced[index])):
                            np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
