from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from .criterion import Criterion
from .optimizer import Optimizer
from core.anomaly.dataloader import DataLoader
from ..state import State


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Engine(object):
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: Optimizer, 
        criterion: Criterion,
    ):
        super().__init__()
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion
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

        # iterable obj
        kwargs = dict(
            iterable=dataloader, 
            desc=f"EPOCH {state.current_epoch}/{state.num_epochs} TRN"
        )

        # start batch loop
        for X in tqdm(**kwargs):
            # to gpu
            X = X.to(DEVICE)

            # forward pass
            with autocast(DEVICE.type):
                X_hat = self.model(X)
                batch_recon = self.criterion(X_hat, X)
                batch_score = batch_recon.mean()

            # backward pass
            self.backprop(batch_score)

            # accumulate loss
            epoch_score += batch_score.item()

        state.trn_score = epoch_score / len(dataloader)

    def backprop(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
