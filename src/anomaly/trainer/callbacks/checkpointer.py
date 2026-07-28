from .base import Callback
import copy


class Checkpointer(Callback):
    def __init__(self):
        super().__init__()
        self.state_dict = None

    def on_epoch_end(self, trainer):
        if trainer.state.is_best:
            self.state_dict = copy.deepcopy(trainer.model.state_dict())

    def end(self, trainer):
        trainer.model.load_state_dict(self.state_dict)