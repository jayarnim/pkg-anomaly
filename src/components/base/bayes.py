from abc import ABC, abstractmethod
from .output import BayesModelOutput
import torch
import torch.nn as nn


class BayesModel(nn.Module, ABC):
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
    ) -> BayesModelOutput:
        ...

    def __call__(self, *args, **kwargs) -> BayesModelOutput:
        outputs = super().__call__(*args, **kwargs)

        if not isinstance(outputs, BayesModelOutput):
            raise TypeError(
                f"{self.__class__.__name__}.forward() "
                f"must return BayesModelOutput, got {type(outputs).__name__}."
            )
        else:
            return outputs