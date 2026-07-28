from ..config.config import *
from .model import *
from .datamodule import *
from .trainer import *
from .evaluator import *


def parser(cfg):
    return Config(
        model=model(cfg),
        datamodule=datamodule(cfg),
        trainer=trainer(cfg),
        evaluator=evaluator(cfg),
        seed=cfg["seed"],
    )