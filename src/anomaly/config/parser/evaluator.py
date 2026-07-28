from ..config.evaluator import *
from core.anomaly.config.parser.criterion import *


def evaluator(cfg):
    return EvaluatorCfg(
        criterion=criterion(cfg),
    )