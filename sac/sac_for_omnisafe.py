from __future__ import annotations

from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo


class MySAC(BaseAlgo):
    def _init(self) -> None:
        """Initialize the algorithm."""

    def _init_env(self) -> None:
        """Initialize the environment."""

    def _init_model(self) -> None:
        """Initialize the model."""

    def _init_log(self) -> None:
        """Initialize the logger."""

    def learn(self) -> tuple[float, float, float]:
        """Learn the policy."""
