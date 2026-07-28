from dataclasses import dataclass
import torch


@dataclass
class BayesModelOutput:
    hat: torch.Tensor
    kld: torch.Tensor