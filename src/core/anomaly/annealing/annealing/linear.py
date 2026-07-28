from .base import Annealing
from .registry import register


@register("linear")
class LinearAnnealing(Annealing):
    def __init__(
        self, 
        min: float, 
        max: float, 
        warmup: int,
    ):
        super().__init__()
        self.min = min
        self.max = max
        self.warmup = warmup

    def __call__(
        self, 
        step: int,
    ) -> float:
        CURRENT = step / self.warmup
        MAX = self.max
        progress = min(CURRENT, MAX)
        beta = self.min + (self.max - self.min) * progress
        return beta