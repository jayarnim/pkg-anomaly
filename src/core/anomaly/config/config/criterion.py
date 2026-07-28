from dataclasses import dataclass


@dataclass
class CriterionCfg:
    name: str
    params: dict