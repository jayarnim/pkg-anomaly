from .criterion import build as build_criterion
from .optimizer import build as build_optimizer
from .engine import Engine
import torch.nn as nn
from ...config.config.trainer import TrnCfg


def build(
    model: nn.Module, 
    cfg: TrnCfg,
) -> Engine:
    kwargs = dict(
        params=model.parameters(),
        cfg=cfg.optimizer,
    )
    optimizer = build_optimizer(**kwargs)

    kwargs = dict(
        cfg=cfg.criterion,
    )
    criterion = build_criterion(**kwargs)

    kwargs = dict(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
    )
    return Engine(**kwargs)