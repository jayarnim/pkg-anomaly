import torch
from torch.utils.data import Dataset
from .registry import register
import pandas as pd


@register("msr")
class MSRDataset(Dataset):
    def __init__(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        **kwargs,
    ):
        super().__init__()
        
        self.X = torch.tensor(
            data=X.values, 
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            data=y.values, 
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(
        self, 
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]