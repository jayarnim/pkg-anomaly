from dataclasses import dataclass
from core.anomaly.config.config.optimizer import *
from core.anomaly.config.config.criterion import *


@dataclass
class TrnCfg:
    optimizer: OptimizerCfg
    criterion: CriterionCfg


@dataclass
class ValCfg:
    criterion: CriterionCfg


@dataclass
class EarlyStoppingCfg:
    patience: int
    delta: float
    warmup: int


@dataclass
class TrainerCfg:
    num_epochs: int
    early_stopping: EarlyStoppingCfg
    trn: TrnCfg
    val: ValCfg