from ..config.trainer import *
from core.anomaly.config.parser.criterion import *
from core.anomaly.config.parser.annealing import *
from core.anomaly.config.parser.optimizer import *
from core.anomaly.config.parser.elbo import *


def trn(cfg):
    return TrnCfg(
        optimizer=optimizer(cfg),
        elbo=elbo(cfg),
    )


def val(cfg):
    return ValCfg(
        elbo=elbo(cfg),
    )


def early_stopping(cfg):
    return EarlyStoppingCfg(
        patience=cfg["early_stopping"]["patience"],
        delta=cfg["early_stopping"]["delta"],
        warmup=cfg["early_stopping"]["warmup"],
    )


def trainer(cfg):
    return TrainerCfg(
        num_epochs=cfg["num_epochs"],
        early_stopping=early_stopping(cfg),
        trn=trn(cfg),
        val=val(cfg),
    )