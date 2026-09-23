from typing import Sequence

import numpy as np

from nn import Parameter
from optim import BaseOptimizer


class Adam(BaseOptimizer):
    def __init__(self,
                 parameters: Sequence[Parameter],
                 lr: float,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8):
        super().__init__(parameters, lr)
        if len(betas) != 2:
            raise ValueError("betas must be a tuple of length 2")
        if not 0 <= betas[0] < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= betas[1] < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self._m = {p: np.zeros_like(p.data) for p in self.parameters}
        self._v = {p: np.zeros_like(p.data) for p in self.parameters}
        self._t = 0
        self.betas = betas
        self.eps = eps


    def step(self):
        self._t += 1

        beta1, beta2 = self.betas

        bias_correction1 = 1 - beta1**self._t
        bias_correction2 = 1 - beta2**self._t

        beta1_complement = 1 - beta1
        beta2_complement = 1 - beta2

        for p in self.parameters:
            grad = p.grad

            if grad is None:
                continue

            self._m[p] = beta1 * self._m[p] + beta1_complement * grad
            self._v[p] = beta2 * self._v[p] + beta2_complement * (grad**2)

            m_hat = self._m[p] / bias_correction1
            v_hat = self._v[p] / bias_correction2

            p.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
