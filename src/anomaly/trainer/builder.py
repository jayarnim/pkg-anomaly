from .trn import build as build_trn
from .val import build as build_val
from .callbacks.earlystopping import EarlyStopping
from .callbacks.logger import Logger
from .callbacks.checkpointer import Checkpointer
from .trainer import Trainer
import torch.nn as nn
from ..config.config.trainer import TrainerCfg


def build(
    model: nn.Module, 
    cfg: TrainerCfg,
) -> Trainer:
    kwargs = dict(
        model=model,
        cfg=cfg.trn,
    )
    trn = build_trn(**kwargs)

    kwargs = dict(
        model=model,
        cfg=cfg.val,
    )
    val = build_val(**kwargs)

    callbacks = [
        EarlyStopping(**vars(cfg.early_stopping)),
        Logger(),
        Checkpointer(),
    ]

    kwargs = dict(
        model=model,
        trn=trn,
        val=val,
        callbacks=callbacks,
        num_epochs=cfg.num_epochs,
    )
    return Trainer(**kwargs)