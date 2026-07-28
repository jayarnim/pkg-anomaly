from dataclasses import dataclass
from core.anomaly.config.config.criterion import *


@dataclass
class EvaluatorCfg:
    criterion: CriterionCfg