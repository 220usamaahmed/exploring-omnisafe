import os
import sys
import warnings

import torch

import omnisafe
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.typing import NamedTuple, Tuple


def train(
    exp_id: str, algo: str, env_id: str, custom_cfgs: NamedTuple
) -> Tuple[float, float, float]:
    """Train a policy from exp-x config with OmniSafe.

    Args:
        exp_id (str): Experiment ID.
        algo (str): Algorithm to train.
        env_id (str): The name of test environment.
        custom_cfgs (NamedTuple): Custom configurations.
        num_threads (int, optional): Number of threads. Defaults to 6.
    """

    terminal_log_name = "terminal.log"
    error_log_name = "error.log"
    if "seed" in custom_cfgs:
        terminal_log_name = f"seed{custom_cfgs['seed']}_{terminal_log_name}"
        error_log_name = f"seed{custom_cfgs['seed']}_{error_log_name}"

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    print(f"exp-x: {exp_id} is training...")

    if not os.path.exists(custom_cfgs["logger_cfgs"]["log_dir"]):
        os.makedirs(custom_cfgs["logger_cfgs"]["log_dir"], exist_ok=True)

    sys.stdout = open(
        os.path.join(f"{custom_cfgs['logger_cfgs']['log_dir']}", terminal_log_name),
        "w",
        encoding="utf-8",
    )
    sys.stderr = open(
        os.path.join(f"{custom_cfgs['logger_cfgs']['log_dir']}", error_log_name),
        "w",
        encoding="utf-8",
    )

    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    reward, cost, ep_len = agent.learn()
    return reward, cost, ep_len


if __name__ == "__main__":
    eg = ExperimentGrid(exp_name="Tutorial_benchmark")

    # Set the algorithms.
    base_policy = [
        "PolicyGradient",
        "NaturalPG",
        "TRPO",
        "PPO",
    ]

    # Set the environments.
    mujoco_envs = [
        "SafetyPointGoal1-v0",
        # "SafetyAntVelocity-v1",
        # "SafetyHopperVelocity-v1",
        # "SafetyHumanoidVelocity-v1",
    ]
    eg.add("env_id", mujoco_envs)
    eg.add("algo", base_policy)
    eg.add("logger_cfgs:use_wandb", [False])
    eg.add("train_cfgs:vector_env_nums", [1])
    eg.add("train_cfgs:torch_threads", [1])
    eg.add("train_cfgs:total_steps", [2048])
    eg.add("algo_cfgs:steps_per_epoch", [1024])
    eg.add("seed", [0])

    eg.run(train, num_pool=1)

    # eg.analyze(
    #     paramter="algo", values=["PPO", "PolicyGradient"], compare_num=None, cost_limit=None
    # )
    # eg.analyze(parameter="algo", values=None, compare_num=3, cost_limit=None)
