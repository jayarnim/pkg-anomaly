from dataclasses import dataclass


@dataclass
class ModelCfg:
    name: str
    params: dict