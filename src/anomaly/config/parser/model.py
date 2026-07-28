from ..config.model import *


def model(cfg):
    return ModelCfg(
        name=cfg["model"]["name"],
        params=cfg["model"].get("params") or dict(),
    )