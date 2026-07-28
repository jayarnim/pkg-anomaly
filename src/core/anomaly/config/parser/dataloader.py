from ..config.dataloader import *


def dataloader(cfg):
    return DataloaderCfg(
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=cfg["dataloader"]["shuffle"],
    )