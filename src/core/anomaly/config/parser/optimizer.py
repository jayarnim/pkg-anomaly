from ..config.optimizer import *


def optimizer(cfg):
    return OptimizerCfg(
        name=cfg["optimizer"]["name"],
        params=cfg["optimizer"].get("params") or dict(),
    )