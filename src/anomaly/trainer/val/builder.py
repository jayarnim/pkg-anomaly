from .criterion import build as build_criterion
from .engine import Engine
import torch.nn as nn
from ...config.config.trainer import ValCfg


def build(
    model: nn.Module, 
    cfg: ValCfg,
) -> Engine:
    kwargs = dict(
        cfg=cfg.criterion,
    )
    criterion = build_criterion(**kwargs)

    kwargs = dict(
        model=model,
        criterion=criterion,
    )
    return Engine(**kwargs)