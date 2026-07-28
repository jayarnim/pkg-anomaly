from .callbacks.base import Callback
from .callbacks.manager import CallbackManager
from .trn import Engine as TrnEngine
from .val import Engine as ValEngine
from .state import State
from .records import Records
import torch
import torch.nn as nn
from core.anomaly.dataloader import DataLoader
from typing import Any


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(
        self, 
        model: nn.Module, 
        trn: TrnEngine, 
        val: ValEngine, 
        callbacks: list[Callback], 
        num_epochs: int,
    ):
        self.model = model.to(DEVICE)
        self.trn = trn
        self.val = val
        self.callbacks = CallbackManager(callbacks=callbacks)
        self.num_epochs = num_epochs
        
        self.state = State(num_epochs=num_epochs)
        self.records = Records()

    def fit(
        self, 
        trn_loader: DataLoader, 
        val_loader: DataLoader, 
    ) -> dict[str, Any]:
        # BEGIN ==========
        self.callbacks.begin(self)

        for epoch in range(self.num_epochs):
            self.state.current_epoch = epoch + 1

            # EPOCH BEGIN ==========
            self.callbacks.on_epoch_begin(self)
            
            # TRN ==========
            self.callbacks.on_trn_begin(self)
            self.trn(trn_loader, self.state)
            self.callbacks.on_trn_end(self)

            # VAL ==========
            self.callbacks.on_val_begin(self)
            self.val(val_loader, self.state)
            self.callbacks.on_val_end(self)

            # EPOCH END ==========
            self.callbacks.on_epoch_end(self)
            self.records.update(self.state)

            # EARLY STOPPING ==========
            if self.state.should_stop==True:
                break

        # END ==========
        self.callbacks.end(self)

        return self.records.get()
