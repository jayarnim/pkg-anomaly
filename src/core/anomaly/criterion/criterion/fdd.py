from .base import Criterion
from .registry import register
import torch
import torch.nn.functional as F


@register("fdd")
class FusedDirectionalDistance(Criterion):
    def __init__(
        self, 
        lamb: float,
    ):
        super().__init__()
        self.lamb = lamb

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
        mse = F.mse_loss(**kwargs).mean(dim=1)

        kwargs = dict(
            x1=pred, 
            x2=true, 
            dim=1,
        )
        dir = F.cosine_similarity(**kwargs)

        return self.lamb * mse + (1-self.lamb) * (1-dir)