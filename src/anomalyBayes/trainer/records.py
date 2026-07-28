from dataclasses import dataclass, field
from typing import Any
from .state import State


@dataclass
class Records:
    trn_nlls: list[float] = field(default_factory=list)
    trn_klds: list[float] = field(default_factory=list)
    val_nlls: list[float] = field(default_factory=list)
    val_klds: list[float] = field(default_factory=list)
    val_recons: list[list] = field(default_factory=list)
    best_epoch: int = 0

    def update(
        self, 
        state: State,
    ) -> None:
        self.trn_nlls.append(state.trn_nll)
        self.trn_klds.append(state.trn_kld)
        self.val_nlls.append(state.val_nll)
        self.val_klds.append(state.val_kld)
        self.val_recons.append(state.val_recon)
        self.best_epoch = state.best_epoch

    def get(self) -> dict[str, Any]:
        trn = dict(
            nll=self.trn_nlls,
            kld=self.trn_klds,
        )
        val = dict(
            nll=self.val_nlls,
            kld=self.val_klds,
        )
        return dict(
            trn=trn,
            val=val,
            best_epoch=self.best_epoch,
            val_anomaly=self.val_recons[self.best_epoch-1],
        )