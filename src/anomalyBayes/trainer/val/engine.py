from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from .elbo import ELBO
from core.anomaly.dataloader import DataLoader
from ..state import State


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Engine(object):
    def __init__(
        self, 
        model: nn.Module, 
        elbo: ELBO,
    ):
        super().__init__()
        self.model = model.to(DEVICE)
        self.elbo = elbo
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
        epoch_nll = 0.0
        epoch_kld = 0.0

        # iterable obj
        kwargs = dict(
            iterable=dataloader, 
            desc=f"EPOCH {state.current_epoch}/{state.num_epochs} VAL"
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

            # accumulate loss
            epoch_recon.append(scores["recon"].detach().cpu())
            epoch_score += scores["score"].item()
            epoch_nll += scores["nll"].item()
            epoch_kld += scores["kld"].item()

        state.val_recon = torch.cat(tensors=epoch_recon, dim=0)
        state.val_score = epoch_score / len(dataloader)
        state.val_nll = epoch_nll / len(dataloader)
        state.val_kld = epoch_kld / len(dataloader)
