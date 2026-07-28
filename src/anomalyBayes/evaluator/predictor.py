from tqdm import tqdm
import torch
import torch.nn as nn
from core.anomaly.dataloader import DataLoader
from .criterion import Criterion


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor(object):
    def __init__(
        self, 
        model: nn.Module, 
        criterion: Criterion,
    ):
        super().__init__()
        self.model = model.to(DEVICE)
        self.criterion = criterion

    @torch.no_grad()
    def __call__(
        self, 
        dataloader: DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # evalidation
        self.model.eval()

        # reset prob & true
        pred_list = []
        true_list = []

        # iterable obj
        kwargs = dict(
            iterable=dataloader, 
            desc=f"TST"
        )

        # start batch loop
        for X, y in tqdm(**kwargs):
            X = X.to(DEVICE)
            outputs = self.model(X)

            recon = self.criterion(outputs.hat, X)
            pred_list.extend(recon.cpu().tolist())
            true_list.extend(y.tolist())

        # list -> tensor
        kwargs = dict(
            data=pred_list, 
            dtype=torch.float32,
        )
        pred_tensor = torch.tensor(**kwargs).squeeze(-1)

        kwargs = dict(
            data=true_list, 
            dtype=torch.int64,
        )
        true_tensor = torch.tensor(**kwargs).squeeze(-1)

        return pred_tensor, true_tensor