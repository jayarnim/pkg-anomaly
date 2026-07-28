class Callback:

    def begin(self, trainer):
        pass

    def on_epoch_begin(self, trainer):
        pass

    def on_trn_begin(self, trainer):
        pass

    def on_trn_end(self, trainer):
        pass

    def on_val_begin(self, trainer):
        pass

    def on_val_end(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def end(self, trainer):
        pass