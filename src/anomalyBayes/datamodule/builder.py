import pandas as pd
from .scaler import scaler
from .split import one_class_split
from .dataloader import build as build_dataloader
from .dataloader import DataLoader
from ..config.config.datamodule import DataModuleCfg


def build(
    df: pd.DataFrame,
    y_col: str,
    cfg: DataModuleCfg,
    scaling: list[str]=None,
) -> dict[str, DataLoader]:
    kwargs = dict(
        df=df,
        y_col=y_col,
        **vars(cfg.split),
    )
    split = one_class_split(**kwargs)

    if scaling is not None:
        split = scaler(
            split=split,
            cols=scaling,
        )

    dataloaders = dict()

    for name, vals in split.items():
        TASK = (
            "msr"
            if name=="tst"
            else "opt"
        )

        kwargs = dict(
            df=vals,
            y_col=y_col,
            task=TASK,
            cfg=cfg.dataloader,
        )
        dataloaders[name] = build_dataloader(**kwargs)

    return dataloaders