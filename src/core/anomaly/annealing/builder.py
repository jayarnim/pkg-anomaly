from . import annealing
from .annealing.registry import ANNEALING_REGISTRY
from .annealing.base import Annealing
from ..config.config.annealing import AnnealingCfg


def build(
    cfg: AnnealingCfg,
) -> Annealing:
    cls = ANNEALING_REGISTRY[cfg.name]
    return cls(**cfg.params)