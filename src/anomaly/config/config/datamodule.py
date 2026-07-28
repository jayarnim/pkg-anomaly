from dataclasses import dataclass
from core.anomaly.config.config.dataloader import *


@dataclass
class SplitCfg:
    ratio: dict[str, int]
    shuffle: bool
    seed: int


@dataclass
class DataModuleCfg:
    split: SplitCfg
    dataloader: DataloaderCfg