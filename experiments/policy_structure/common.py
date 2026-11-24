"""Checkpoint, actuator, and artifact handling shared by the experiments."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_desired_thrust import DesiredThrustExtension, direct_wrench_action
from envs.quadrotor_smooth import ALLOCATION, VARIANTS, Reference, State, at, make_environment
from envs.quadrotor_thrust_rate import MAX_THRUST, TORQUE_LIMITS, ThrustRateInputs, ThrustState
from experiments.policy_structure.lee import Gains, lee_terms, lee_wrench
from experiments.quadrotor_comparison import load_policy, write_json
from rl.quadrotor_ppo import command


ACTION_NAMES = ("f", "Mx", "My", "Mz")
ACTION_UNITS = ("N", "N m", "N m", "N m")
ACTION_SCALES = np.array([MAX_THRUST, *map(float, TORQUE_LIMITS)])
SPLITS = ("train", "validation", "test")
SCHEMA_VERSION = 1
DYNAMIC_THRUST = {
    "H_dynamic_thrust_extension",
    "I_full_double_thrust_extension",
    "J_symmetry_double_thrust_extension",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_hashes(directory):
    return {name: digest(directory / name) for name in ("config.json", "policy.msgpack")}


@dataclass
class Policy:
    directory: Path
    model: object
    params: object
    config: dict

    @classmethod
    def load(cls, directory):
        directory = Path(directory).resolve()
        config_path = directory / "config.json"
        if not config_path.is_file() or not (directory / "policy.msgpack").is_file():
            raise ValueError("Provide a smooth quadrotor checkpoint directory containing config.json and policy.msgpack")
        model, params, config = load_policy(directory)
        variant = config["variant"]
        if variant not in VARIANTS or variant in DYNAMIC_THRUST:
            raise ValueError(f"{variant} requires a different plant or thrust-state analysis")
        return cls(directory, model, params, config)

    def environment(self, bank, dt):
        env = make_environment(bank, self.config["variant"], dt=dt,
                               reward_profile=self.config.get("reward_profile", "legacy"))
        if env.obs_size != self.config["observation_size"]:
            raise ValueError("Checkpoint observation size disagrees with the environment")
        scale = np.asarray(self.config.get("observation_scale", env.scale))
        if scale.shape != (env.obs_size,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Invalid checkpoint observation scales")
        env.scale = jnp.asarray(scale, dtype=env.scale.dtype)
        return env

    def from_observation(self, normalized_obs):
        mean, _, _ = self.model.apply(self.params, normalized_obs)
        return command(mean)

    def controller(self, env):
        return lambda state: self.from_observation(env.observe(state))


def actor_wrench(env, native_command):
    """Applied wrench, with the exact allocator and clipping used by transition."""
    if isinstance(env, DesiredThrustExtension):
        native_command = direct_wrench_action(native_command)[0]
    return ALLOCATION @ jnp.clip(native_command, 0., 1.)


def actor_requested_wrench(env, native_command):
    """Physical actor command after tanh, before the direct-wrench torque allocator."""
    channels = jnp.clip(native_command, 0., 1.)
    if isinstance(env, DesiredThrustExtension):
        return jnp.concatenate((MAX_THRUST * channels[:1], TORQUE_LIMITS * (2 * channels[1:] - 1)))
    return ALLOCATION @ channels


def wrench_channels(requested):
    return jnp.concatenate((requested[:1] / MAX_THRUST,
                            (requested[1:] / TORQUE_LIMITS + 1.) / 2.))


def feasible_wrench(requested):
    """Thrust-preserving radial allocation using the existing direct-wrench decoder."""
    rotors = direct_wrench_action(wrench_channels(requested))[0]
    return ALLOCATION @ jnp.clip(rotors, 0., 1.)


def wrench_controller(env, request):
    def controller(state):
        channels = wrench_channels(request(state))
        if isinstance(env, DesiredThrustExtension):
            return channels
        return direct_wrench_action(channels)[0]
    return controller


def lee_controller(env, gains):
    def request(state):
        ref = at(env.bank, state.trajectory, state.step)
        return lee_wrench(lee_terms(state, ref, gains), gains)
    return request


def take(tree, indices):
    return jax.tree_util.tree_map(lambda values: values[indices], tree)


def concatenate(trees):
    return jax.tree_util.tree_map(lambda *values: jnp.concatenate(values), *trees)


@dataclass
class Dataset:
    directory: Path
    policy: Policy
    env: object
    states: tuple
    rows: dict
    split: np.ndarray
    family: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, directory, checkpoint=None):
        directory = Path(directory).resolve()
        metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if metadata["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported dataset schema")
        policy = Policy.load(checkpoint or metadata["checkpoint"])
        if checkpoint_hashes(policy.directory) != metadata["checkpoint_sha256"]:
            raise ValueError("Checkpoint does not match the saved policy targets")
        for name in ("references.npz", "samples.npz"):
            if digest(directory / name) != metadata["artifacts_sha256"][name]:
                raise ValueError(f"Saved dataset artifact failed its hash check: {name}")
        with np.load(directory / "references.npz", allow_pickle=False) as arrays:
            bank = Reference(*(jnp.asarray(arrays[name]) for name in Reference._fields))
            split, family = arrays["split"], arrays["family"]
        env = policy.environment(bank, metadata["dt"])
        state_type = ThrustState if isinstance(env, ThrustRateInputs) else State
        with np.load(directory / "samples.npz", allow_pickle=False) as arrays:
            states = state_type(*(jnp.asarray(arrays[name]) for name in state_type._fields))
            rows = dict(arrays)
        return cls(directory, policy, env, states, rows, split, family, metadata)

    @property
    def gains(self):
        return Gains(**self.metadata["lee_gains"])

    @property
    def sample_split(self):
        return self.split[self.rows["trajectory"]]

    @property
    def sample_family(self):
        return self.family[self.rows["trajectory"]]

    @property
    def sample_source(self):
        return self.rows["source"]

    def groups(self):
        """Report each split overall and across its families and state sources."""
        for split in SPLITS:
            in_split = self.sample_split == split
            for family in ("all", *np.unique(self.sample_family[in_split])):
                in_family = in_split if family == "all" else in_split & (self.sample_family == family)
                for source in ("all", *np.unique(self.sample_source[in_family])):
                    indices = np.flatnonzero(in_family if source == "all" else
                                             in_family & (self.sample_source == source))
                    if len(indices):
                        yield split, family, source, indices

    def analysis_manifest(self, output, **settings):
        write_json(output / "manifest.json", {
            "dataset": str(self.directory),
            "dataset_manifest_sha256": digest(self.directory / "manifest.json"),
            "checkpoint_sha256": self.metadata["checkpoint_sha256"],
            "analysis_source_sha256": {path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
            **settings,
        })


def experiment_arguments(parser):
    parser.add_argument("--dataset", type=Path, required=True, help="Directory generated by policy_structure.data")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--checkpoint", type=Path, help="Location override for the same checkpoint")
