from dataclasses import dataclass


@dataclass
class AnnealingCfg:
    name: str
    params: dict