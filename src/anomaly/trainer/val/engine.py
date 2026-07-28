from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from .criterion import Criterion
from core.anomaly.dataloader import DataLoader
from ..state import State


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Engine(object):
    def __init__(
        self, 
        model: nn.Module, 
        criterion: Criterion,
    ):
        super().__init__()
        self.model = model.to(DEVICE)
        self.criterion = criterion
        self.scaler = GradScaler(device=DEVICE)

    @torch.no_grad()
    def __call__(
        self, 
        dataloader: DataLoader, 
        state: State,
    ) -> None:
        # train
        self.model.eval()

        # reset epoch recon & loss
        epoch_recon = []
        epoch_score = 0.0

        # iterable obj
        kwargs = dict(
            iterable=dataloader, 
            desc=f"EPOCH {state.current_epoch}/{state.num_epochs} VAL"
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

            # accumulate loss
            epoch_recon.append(batch_recon.detach().cpu())
            epoch_score += batch_score.item()

        state.val_recon = torch.cat(tensors=epoch_recon, dim=0)
        state.val_score = epoch_score / len(dataloader)

