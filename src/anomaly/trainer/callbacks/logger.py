from .base import Callback
from IPython.display import clear_output


class Logger(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, trainer):
        print(
            f'TRN SCORE: {trainer.state.trn_score:.4f}',
        )
        print(
            f'VAL SCORE: {trainer.state.val_score:.4f}',
            f'COUNTER: {trainer.state.counter}',
            sep='\t\t',
        )
        if (trainer.state.current_epoch) % 50 == 0:
            clear_output(wait=False)

    def end(self, trainer):
        clear_output(wait=False)

        print(
            f"BEST EPOCH: {trainer.state.best_epoch}",
            f"BEST SCORE: {trainer.state.best_score:.4f}",
            sep="\n",
        )