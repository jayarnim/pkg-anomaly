import pandas as pd
import torch
from .predictor import Predictor
from .calculator import Calculator
from core.anomaly.dataloader import DataLoader


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator(object):
    def __init__(
        self, 
        predictor: Predictor, 
        calculator: Calculator,
    ):
        super().__init__()
        self.predictor = predictor
        self.calculator = calculator

    def __call__(
        self, 
        dataloader: DataLoader,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        kwargs = dict(
            dataloader=dataloader,
        )
        pred, true = self.predictor(**kwargs)

        data = dict(
            pred=pred.cpu().numpy(),
            true=true.cpu().numpy(),
        )
        result = pd.DataFrame(data)

        kwargs = dict(
            pred=pred,
            true=true,
        )
        metrics_sheet = self.calculator(**kwargs)

        return result, metrics_sheet