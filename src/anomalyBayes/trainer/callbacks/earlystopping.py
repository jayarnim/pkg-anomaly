from .base import Callback


class EarlyStopping(Callback):
    def __init__(self, patience, delta, warmup):
        super().__init__()
        self.patience = patience
        self.delta = delta
        self.warmup = warmup

    def on_val_end(self, trainer):
        CURRENT_EPOCH = trainer.state.current_epoch
        CURRENT_SCORE = trainer.state.val_score
        REFERENCE = trainer.state.best_score - self.delta
        IMPROVED = CURRENT_SCORE < REFERENCE

        if CURRENT_EPOCH <= self.warmup:
            trainer.state.is_best = False
            trainer.state.counter = 0
            trainer.state.should_stop = False

        else:
            if IMPROVED:
                trainer.state.is_best = True
                trainer.state.best_epoch = CURRENT_EPOCH
                trainer.state.best_score = CURRENT_SCORE
                trainer.state.counter = 0
                trainer.state.should_stop = False
        
            else:
                trainer.state.is_best = False
                trainer.state.counter += 1

        if trainer.state.counter > self.patience:
            trainer.state.should_stop = True