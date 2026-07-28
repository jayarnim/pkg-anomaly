from ..config.elbo import *
from .criterion import *
from .annealing import *


def elbo(cfg):
    return ELBOCfg(
        criterion=criterion(cfg),
        annealing=annealing(cfg),
    )