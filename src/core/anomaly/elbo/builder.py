from .criterion import build as build_criterion
from .annealing import build as build_annealing
from .elbo import ELBO
from ..config.config.elbo import ELBOCfg


def build(
    cfg: ELBOCfg,
) -> ELBO:
    kwargs = dict(
        cfg=cfg.criterion,
    )
    criterion = build_criterion(**kwargs)

    kwargs = dict(
        cfg=cfg.annealing,
    )
    annealing = build_annealing(**kwargs)

    kwargs = dict(
        criterion=criterion,
        annealing=annealing,
    )
    return ELBO(**kwargs)