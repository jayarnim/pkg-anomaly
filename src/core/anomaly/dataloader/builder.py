from typing import Literal
import pandas as pd
from . import dataset
from .dataset.registry import DATASET_REGISTRY
from torch.utils.data import DataLoader
from ..config.config.dataloader import DataloaderCfg


def build(
    df: pd.DataFrame, 
    y_col: str, 
    task: Literal["opt", "msr"],
    cfg: DataloaderCfg,
) -> DataLoader:
    X_col = df.columns.difference([y_col])
    
    kwargs = dict(
        X=df.loc[:, X_col],
        y=df.loc[:, [y_col]],
    )
    dataset = DATASET_REGISTRY[task](**kwargs)

    kwargs = dict(
        dataset=dataset, 
        batch_size=cfg.batch_size, 
        shuffle=cfg.shuffle,           
    )
    return DataLoader(**kwargs)