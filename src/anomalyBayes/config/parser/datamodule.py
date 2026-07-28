from ..config.datamodule import *
from core.anomaly.config.parser.dataloader import *


def split(cfg):
    return SplitCfg(
        ratio=cfg["split"]["ratio"],
        shuffle=cfg["split"]["shuffle"],
        seed=cfg["seed"],
    )


def datamodule(cfg):
    return DataModuleCfg(
        split=split(cfg),
        dataloader=dataloader(cfg),
    )