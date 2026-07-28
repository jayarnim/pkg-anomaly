from dataclasses import dataclass
from .datamodule import *
from .trainer import *
from .evaluator import *
from .model import *


@dataclass
class Config:
    model: ModelCfg
    datamodule: DataModuleCfg
    trainer: TrainerCfg
    evaluator: EvaluatorCfg
    seed: int