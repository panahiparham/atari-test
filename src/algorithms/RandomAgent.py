import numpy as np

from typing import Any, Dict, Tuple
from PyExpUtils.collection.Collector import Collector

from algorithms.BaseAgent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

    def start(self, x: np.ndarray):
        return self.rng.integers(self.actions)

    def step(self, r: float, xp: np.ndarray | None, extra: Dict[str, Any]):
        return self.rng.integers(self.actions)

    def end(self, r: float, extra: Dict[str, Any]):
        ...
