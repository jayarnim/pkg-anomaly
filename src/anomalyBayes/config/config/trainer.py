from dataclasses import dataclass
from core.anomaly.config.config.criterion import *
from core.anomaly.config.config.annealing import *
from core.anomaly.config.config.optimizer import *
from core.anomaly.config.config.elbo import *


@dataclass
class TrnCfg:
    optimizer: OptimizerCfg
    elbo: ELBOCfg


@dataclass
class ValCfg:
    elbo: ELBOCfg


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