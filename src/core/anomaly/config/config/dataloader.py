from dataclasses import dataclass


@dataclass
class DataloaderCfg:
    batch_size: int
    shuffle: bool