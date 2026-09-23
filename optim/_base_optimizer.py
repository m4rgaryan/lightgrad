from typing import Sequence

from abc import ABC, abstractmethod

from nn import Parameter


class BaseOptimizer(ABC):
    def __init__(self, parameters: Sequence[Parameter], lr: float):
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.learning_rate = lr
        self.parameters = list(parameters)

    @abstractmethod
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()