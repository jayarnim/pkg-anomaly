from .predictor import Predictor
from .calculator import Calculator
from .evaluator import Evaluator
from .criterion import build as build_criterion
import torch
import torch.nn as nn
from ..config.config.evaluator import EvaluatorCfg


def build(
    model: nn.Module, 
    val_anomaly: torch.Tensor, 
    percentiles: list[float], 
    cfg: EvaluatorCfg,
) -> Evaluator:
    kwargs = dict(
        cfg=cfg.criterion,
    )
    criterion = build_criterion(**kwargs)

    kwargs = dict(
        model=model,
        criterion=criterion,
    )
    predictor = Predictor(**kwargs)

    kwargs = dict(
        val_anomaly=val_anomaly,
        percentiles=percentiles,
    )
    calculator = Calculator(**kwargs)

    kwargs = dict(
        predictor=predictor,
        calculator=calculator,
    )
    return Evaluator(**kwargs)