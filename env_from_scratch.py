from __future__ import annotations

import random
import omnisafe
from typing import Any, ClassVar

import torch
from gymnasium import spaces

from omnisafe.envs.core import CMDP, env_register, env_unregister

@env_register
@env_unregister
class ExampleEnv(CMDP):
    _support_envs: ClassVar[list[str]] = ['Example-v0', 'Example-v1']

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    def __init__(self, env_id: str, **kwargs) -> None:
        self._count = 0
        self._num_envs = 1
        self._observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if seed is not None:
            self.set_seed(seed)
        obs = torch.as_tensor(self._observation_space.sample())
        self._count = 0
        return obs, {}

    @property
    def max_episode_steps(self) -> None:
        return 10

    def render(self) -> Any:
        ...

    def close(self) -> None:
        ... 

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._count += 1
        obs = torch.as_tensor(self._observation_space.sample())
        reward = 2 * torch.as_tensor(random.random())
        cost = 2 * torch.as_tensor(random.random())
        terminated = torch.as_tensor(random.random() > 0.9)
        truncated = torch.as_tensor(self._count > 10)
        return obs, reward, cost,terminated, truncated, {'final_observation': obs}


@env_register
@env_unregister
class NewExampleEnv(ExampleEnv):
    _support_envs: ClassVar[list[str]] = ['NewExample-v0', 'NewExample-v1']
    num_agents: ClassVar[int] = 1

    def __init__(self, env_id: str, **kwargs) -> None:
        super(NewExampleEnv, self).__init__(env_id, **kwargs)
        self.env_spec_log = {"Env/Success_counts": 0}
        self.num_agents = kwargs.get('num_agents', 1)

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = super().step(action)
        success = int(reward > cost)
        self.env_spec_log['Env/Success_counts'] += success
        return obs, reward, cost, terminated, truncated, info

    def spec_log(self, logger) -> dict[str, Any]:
        logger.store({'Env/Success_counts': self.env_spec_log['Env/Success_counts']})
        self.env_spec_log['Env/Success_counts'] = 0


def run_env():
    env = ExampleEnv(env_id="Example-v0")
    env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, cost, terminated, truncated, info = env.step(action)
        print('-' * 20)
        print(f'obs: {obs}')
        print(f'reward: {reward}')
        print(f'cost: {cost}')
        print(f'terminated: {terminated}')
        print(f'truncated: {truncated}')
        print(f'info: {info}')
        print('*' * 20)

        if terminated or truncated:
            break

    env.close()

def train_agent():
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 30,
        },
        'algo_cfgs': {
            'steps_per_epoch': 10,
            'update_iters': 1,
        },
    }

    agent = omnisafe.Agent('PPOLag', 'Example-v0', custom_cfgs=custom_cfgs)
    agent.learn()

    custom_cfgs.update({'env_cfgs': {'num_agents': 2}})
    agent = omnisafe.Agent('PPOLag', 'NewExample-v0', custom_cfgs=custom_cfgs)
    print("Num Agents:", agent.agent._env._env.num_agents)


if __name__ == "__main__":
    run_env()
    train_agent()
