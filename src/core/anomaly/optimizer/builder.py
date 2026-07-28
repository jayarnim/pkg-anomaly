from . import optimizer
from .optimizer.registry import OPTIMIZER_REGISTRY
from collections.abc import Iterator
from torch.nn import Parameter
from torch.optim import Optimizer
from ..config.config.optimizer import OptimizerCfg


def build(
    params: Iterator[Parameter], 
    cfg: OptimizerCfg,
) -> Optimizer:
    kwargs = dict(
        params=params, 
        **cfg.params,
    )
    cls = OPTIMIZER_REGISTRY[cfg.name]
    return cls(**kwargs)