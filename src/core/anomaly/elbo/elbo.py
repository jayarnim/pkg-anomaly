from .annealing import Annealing
from .criterion import Criterion
from components.base import BayesModelOutput
import torch


class ELBO(object):
    def __init__(
        self, 
        criterion: Criterion, 
        annealing: Annealing,
    ):
        super().__init__()
        self.criterion = criterion
        self.annealing = annealing

    def __call__(
        self, 
        output: BayesModelOutput, 
        X: torch.Tensor, 
        step: int,
    ) -> dict[str, torch.Tensor]:
        recon = self.criterion(output.hat, X)
        nll = recon.mean()
        kld = output.kld.mean()
        beta = self.annealing(step)
        score = nll + beta * kld

        return dict(
            recon=recon,
            score=score,
            nll=nll,
            kld=kld,
        )