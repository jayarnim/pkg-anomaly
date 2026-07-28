from . import criterion
from .criterion.registry import CRITERION_REGISTRY
from .criterion.base import Criterion
from ..config.config.criterion import CriterionCfg


def build(
    cfg: CriterionCfg,
) -> Criterion:
    cls = CRITERION_REGISTRY[cfg.name]
    return cls(**cfg.params)