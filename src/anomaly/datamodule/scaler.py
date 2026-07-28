import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scaler(
    split: dict[str, pd.DataFrame], 
    cols: list[str],
) -> dict[str, pd.DataFrame]:
    trn = split["trn"]
    val = split["val"]
    tst = split["tst"]

    for df in [trn, val, tst]:
        df.loc[:,cols] = np.log1p(df.loc[:,cols])

    scaler = StandardScaler()
    scaler.fit(trn[cols])

    for df in [trn, val, tst]:
        df.loc[:,cols] = scaler.transform(df.loc[:,cols])

    return dict(
        trn=trn,
        val=val,
        tst=tst,
    )