from abc import ABC, abstractmethod
import torch


class Criterion(ABC):
    @abstractmethod
    def __call__(
        self, 
        pred: torch.Tensor, 
        true: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError