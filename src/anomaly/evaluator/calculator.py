import pandas as pd
import torch
from torchmetrics.classification import (
    BinaryAccuracy, 
    BinaryPrecision, 
    BinaryRecall,
    BinaryF1Score, 
    BinaryConfusionMatrix
)


class Calculator(object):
    def __init__(
        self, 
        val_anomaly: torch.Tensor, 
        percentiles: list[float],
    ):
        super().__init__()
        self.val_anomaly = val_anomaly
        self.percentiles = percentiles
        self.confmat = BinaryConfusionMatrix()
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1 = BinaryF1Score()

    def __call__(
        self, 
        pred: torch.Tensor, 
        true: torch.Tensor,
    ) -> pd.DataFrame:
        metrics = [
            self.calc(pred, true, p)
            for p in self.percentiles
        ]
        return pd.DataFrame(metrics)

    def calc(self, pred, true, p):
        self.confmat.reset()
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

        kwargs = dict(
            input=self.val_anomaly, 
            q=p,
        )
        threshold = torch.quantile(**kwargs).cpu()

        args = ((pred > threshold).int(), true)

        confmat = self.confmat(*args).cpu().numpy()

        return dict(
            p=p,
            threshold=threshold.item(),
            tp=confmat[1,1],
            tn=confmat[0,0],
            fp=confmat[0,1],
            fn=confmat[1,0],
            accuracy=self.accuracy(*args).item(),
            precision=self.precision(*args).item(),
            recall=self.recall(*args).item(),
            f1=self.f1(*args).item(),
        )