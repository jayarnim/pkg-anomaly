class CallbackManager(object):
    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = callbacks

    def begin(self, trainer):
        for cb in self.callbacks:
            cb.begin(trainer)

    def end(self, trainer):
        for cb in self.callbacks:
            cb.end(trainer)

    def on_epoch_begin(self, trainer):
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer)

    def on_epoch_end(self, trainer):
        for cb in self.callbacks:
            cb.on_epoch_end(trainer)

    def on_trn_begin(self, trainer):
        for cb in self.callbacks:
            cb.on_trn_begin(trainer)

    def on_trn_end(self, trainer):
        for cb in self.callbacks:
            cb.on_trn_end(trainer)

    def on_val_begin(self, trainer):
        for cb in self.callbacks:
            cb.on_val_begin(trainer)

    def on_val_end(self, trainer):
        for cb in self.callbacks:
            cb.on_val_end(trainer)
