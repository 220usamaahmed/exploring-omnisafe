from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn

from omnisafe.adapter import OffPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.commen.logger import Logger


@registry.register
class DQN(BaseAlgo):
    def _init_env(self) -> None: ...

    def _init_model(self) -> None: ...

    def _init(self) -> None: ...

    def _init_log(self) -> None: ...

    def learn(self) -> tuple[float, float, float]: ...
