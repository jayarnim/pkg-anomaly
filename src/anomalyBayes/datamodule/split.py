import pandas as pd
from sklearn.model_selection import train_test_split


def one_class_split(
    df: pd.DataFrame, 
    y_col: str,
    ratio: dict,
    shuffle: bool,
    seed: int,
) -> dict[str, pd.DataFrame]:
    normal = df[df[y_col]==0]
    
    tst_abnormal = df[df["Class"]==1]

    kwargs = dict(
        n=len(tst_abnormal),
        random_state=seed,
    )
    tst_normal = normal.sample(**kwargs)

    kwargs = dict(
        objs=[tst_normal, tst_abnormal],
        ignore_index=True,
    )
    tst = pd.concat(**kwargs)

    opt_normal = normal.drop(tst_normal.index)

    kwargs = dict(
        train_size=ratio["trn"],
        test_size=ratio["val"], 
        shuffle=shuffle,
        random_state=seed,
    )
    trn, val = train_test_split(opt_normal, **kwargs)

    return dict(
        trn=trn, 
        val=val, 
        tst=tst,
    )