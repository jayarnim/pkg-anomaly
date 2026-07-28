from .elbo import build as build_elbo
from .engine import Engine
import torch.nn as nn
from ...config.config.trainer import ValCfg


def build(
    model: nn.Module, 
    cfg: ValCfg,
) -> Engine:
    kwargs = dict(
        cfg=cfg.elbo,
    )
    elbo = build_elbo(**kwargs)

    kwargs = dict(
        model=model,
        elbo=elbo,
    )
    return Engine(**kwargs)