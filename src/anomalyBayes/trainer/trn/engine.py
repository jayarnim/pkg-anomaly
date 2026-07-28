from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from .optimizer import Optimizer
from .elbo import ELBO
from core.anomaly.dataloader import DataLoader
from ..state import State


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Engine(object):
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: Optimizer, 
        elbo: ELBO,
    ):
        super().__init__()
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.elbo = elbo
        self.scaler = GradScaler(device=DEVICE)

    def __call__(
        self, 
        dataloader: DataLoader, 
        state: State,
    ) -> None:
        # train
        self.model.train()

        # reset epoch loss
        epoch_score = 0.0
        epoch_nll = 0.0
        epoch_kld = 0.0

        # iterable obj
        kwargs = dict(
            iterable=dataloader, 
            desc=f"EPOCH {state.current_epoch}/{state.num_epochs} TRN"
        )

        # start batch loop
        for X in tqdm(**kwargs):
            # to gpu
            X=X.to(DEVICE)

            # forward pass
            with autocast(DEVICE.type):
                output = self.model(X)

                kwargs = dict(
                    output=output,
                    X=X,
                    step=state.current_epoch,
                )
                scores = self.elbo(**kwargs)

            # backward pass
            self.backprop(scores["score"])

            # accumulate loss
            epoch_score += scores["score"].item()
            epoch_nll += scores["nll"].item()
            epoch_kld += scores["kld"].item()

        state.trn_score = epoch_score / len(dataloader)
        state.trn_nll = epoch_nll / len(dataloader)
        state.trn_kld = epoch_kld / len(dataloader)

    def backprop(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
