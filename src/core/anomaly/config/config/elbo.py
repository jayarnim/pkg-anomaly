from dataclasses import dataclass
from .criterion import *
from .annealing import *


@dataclass
class ELBOCfg:
    criterion: CriterionCfg
    annealing: AnnealingCfg