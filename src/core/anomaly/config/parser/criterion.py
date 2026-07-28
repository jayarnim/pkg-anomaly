from ..config.criterion import *


def criterion(cfg):
    return CriterionCfg(
        name=cfg["criterion"]["name"],
        params=cfg["criterion"].get("params") or dict(),
    )