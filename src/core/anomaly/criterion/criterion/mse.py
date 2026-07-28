from .base import Criterion
from .registry import register
import torch
import torch.nn.functional as F


@register("mse")
class MeanSquaredError(Criterion):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(
        self, 
        pred: torch.Tensor, 
        true: torch.Tensor,
    ) -> torch.Tensor:
        kwargs = dict(
            input=pred, 
            target=true, 
            reduction='none',
        )
        return F.mse_loss(**kwargs).mean(dim=1)