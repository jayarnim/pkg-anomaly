from dataclasses import dataclass


@dataclass
class OptimizerCfg:
    name: str
    params: dict