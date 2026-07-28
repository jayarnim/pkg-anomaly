from ..config.annealing import *


def annealing(cfg):
    return AnnealingCfg(
        name=cfg["annealing"]["name"],
        params=cfg["annealing"].get("params") or dict(),
    )