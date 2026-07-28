from abc import abstractmethod
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, kwargs):
        super().__init__()

        init_args = kwargs.copy()
        init_args.pop("self", None)
        init_args.pop("__class__", None)

        self.init_args = init_args

    @abstractmethod
    def forward(
        self, 
        X: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError