from dataclasses import dataclass, field
from typing import Any
from .state import State


@dataclass
class Records:
    trn_scores: list[float] = field(default_factory=list)
    val_scores: list[float] = field(default_factory=list)
    val_recons: list[list] = field(default_factory=list)
    best_epoch: int = 0

    def update(
        self, 
        state: State,
    ) -> None:
        self.trn_scores.append(state.trn_score)
        self.val_scores.append(state.val_score)
        self.val_recons.append(state.val_recon)
        self.best_epoch = state.best_epoch

    def get(self) -> dict[str, Any]:
        return dict(
            trn=self.trn_scores,
            val=self.val_scores,
            best_epoch=self.best_epoch,
            val_anomaly=self.val_recons[self.best_epoch-1],
        )