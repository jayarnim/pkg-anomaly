import torch
from torch.utils.data import Dataset
from .registry import register
import pandas as pd


@register("opt")
class OPTDataset(Dataset):
    def __init__(
        self, 
        X: pd.DataFrame,
        **kwargs,
    ):
        super().__init__()
        
        self.X = torch.tensor(
            data=X.values, 
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(
        self, 
        idx: int,
    ) -> torch.Tensor:
        return self.X[idx]