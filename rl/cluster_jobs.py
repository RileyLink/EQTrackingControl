"""The eight matched, from-scratch symmetry training jobs (stdlib only)."""

from dataclasses import dataclass


CAMPAIGN_VERSION = 2
ENCODER_VERSION = "actual_body_desired_minus_actual_v1"


@dataclass(frozen=True)
class Job:
    index: int
    robot: str
    label: str
    mode: int
    inputs: int

    def name(self, seed):
        return f"{self.robot}_{self.label}_seed{seed}"


JOBS = tuple(Job(index, robot, label, mode, inputs)
             for index, (robot, label, mode, inputs) in enumerate([
                 ("quadrotor", "full", 0, 40), ("quadrotor", "strong", 1, 31),
                 ("quadrotor", "corrected_strong", 3, 31), ("quadrotor", "weak", 2, 25),
                 ("astrobee", "full", 0, 42), ("astrobee", "strong", 1, 30),
                 ("astrobee", "corrected_strong", 6, 30), ("astrobee", "weak", 2, 27),
             ]))


def job_config(index, seed=0, smoke=False):
    """Use the last upstream SLURM presets, with the attitude coefficient wired.

    Within a robot only the observation mode and experiment name differ.
    Full and smoke campaigns must use separate output directories.
    """
    if index < 0 or index >= len(JOBS) or seed < 0:
        raise ValueError("Task index must be 0..7 and seed must be nonnegative.")
    job = JOBS[index]
    config = dict(
        env_name=f"{job.robot}_random_walk", seed=seed, num_seeds=1,
        EXP_NAME=job.name(seed), EQUIVARIANT=job.mode, SYMMETRY_TYPE=2,
        NUM_LAYERS=6, NUM_NODES=128 if job.robot == "quadrotor" else 256,
        ACTIVATION="leaky_relu", OUT_ACTIVATION="clipped_relu" if job.robot == "quadrotor" else "hard_tanh",
        TOTAL_TIMESTEPS=1_000_000_000 if job.robot == "quadrotor" else 400_000_000,
        NUM_ENVS=64 if job.robot == "quadrotor" else 32, NUM_STEPS=512,
        UPDATE_EPOCHS=5, NUM_MINIBATCHES=4, LR=3e-4, GAMMA=.995, LAMBDA=.95,
        CLIP_RANGE=.2, ENT_COEF=0., VF_COEF=.5, MAX_GRAD_NORM=.5, ANNEAL_LR=True,
        TERMINATE_ON_ERROR=True, TERMINATION_BOUND=10. if job.robot == "quadrotor" else 25.,
        TERMINAL_REWARD=-25., STATE_COV_SCALAR=.5, REF_COV_SCALAR=3.,
        REWARD_Q_POS=.02 if job.robot == "quadrotor" else .05, REWARD_Q_VEL=.001,
        REWARD_Q_ROTM=.6 if job.robot == "quadrotor" else 10.,
        REWARD_Q_OMEGA=.0001 if job.robot == "quadrotor" else .05,
        REWARD_R=.0005 if job.robot == "quadrotor" else .001, REWARD_REACH=True,
        USE_DES_ACTION_IN_REWARD=True, USE_ABS_REWARD_FN=False, REWARD_FN_TYPE=1,
        CLIP_ACTIONS=True, DEBUG=False, COMPACT_METRICS=True,
        CAMPAIGN_VERSION=CAMPAIGN_VERSION, ENCODER_VERSION=ENCODER_VERSION, SMOKE_TEST=smoke,
        REWARD_IMPLEMENTATION="attitude_weight_wired_v2",
        OUTPUT_IMPLEMENTATION="clipped_relu_0_1_v1" if job.robot == "quadrotor" else "flax_hard_tanh",
    )
    if smoke:
        config.update(TOTAL_TIMESTEPS=64, NUM_ENVS=2, NUM_STEPS=16,
                      UPDATE_EPOCHS=1, NUM_MINIBATCHES=2)
    threshold = 400_000_000 if job.robot == "quadrotor" else 20_000_000
    batch = config["NUM_ENVS"] * config["NUM_STEPS"]
    config.update(NUM_UPDATES=config["TOTAL_TIMESTEPS"] // batch,
                  MINIBATCH_SIZE=batch // config["NUM_MINIBATCHES"],
                  LR_NUM_UPDATES=threshold // batch)
    return config
