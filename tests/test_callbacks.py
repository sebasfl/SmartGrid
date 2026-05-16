import pytest

pytest.importorskip("tensorflow")

from src.training.callbacks import EarlyStopping, ReduceLROnPlateau


class MockOptimizer:
    """Minimal optimizer mock for testing callbacks."""
    def __init__(self, lr=0.001):
        self._lr = lr
        self.learning_rate = type('LR', (), {
            'assign': lambda self_inner, val: setattr(self_inner, '_val', val),
            '__float__': lambda self_inner: self_inner._val,
        })()
        self.learning_rate._val = lr


class TestEarlyStopping:
    def test_triggers_after_patience(self):
        es = EarlyStopping(monitor='val_loss', patience=3, verbose=False)
        es.on_train_begin()

        # Epoch 0: good loss
        assert not es.on_epoch_end(0, {'val_loss': 1.0}, None, None)
        # Epochs 1-3: no improvement
        assert not es.on_epoch_end(1, {'val_loss': 1.0}, None, None)
        assert not es.on_epoch_end(2, {'val_loss': 1.0}, None, None)
        # Epoch 3: patience=3, should stop
        assert es.on_epoch_end(3, {'val_loss': 1.0}, None, None)

    def test_does_not_trigger_when_improving(self):
        es = EarlyStopping(monitor='val_loss', patience=3, verbose=False)
        es.on_train_begin()

        for epoch in range(10):
            loss = 1.0 - epoch * 0.01
            should_stop = es.on_epoch_end(epoch, {'val_loss': loss}, None, None)
            assert not should_stop

    def test_resets_on_improvement(self):
        es = EarlyStopping(monitor='val_loss', patience=3, verbose=False)
        es.on_train_begin()

        es.on_epoch_end(0, {'val_loss': 1.0}, None, None)
        es.on_epoch_end(1, {'val_loss': 1.0}, None, None)  # wait=1
        es.on_epoch_end(2, {'val_loss': 1.0}, None, None)  # wait=2
        es.on_epoch_end(3, {'val_loss': 0.5}, None, None)  # improvement! wait=0
        assert not es.on_epoch_end(4, {'val_loss': 0.5}, None, None)  # wait=1

    def test_missing_metric_returns_false(self):
        es = EarlyStopping(monitor='val_loss', patience=3, verbose=False)
        es.on_train_begin()
        assert not es.on_epoch_end(0, {'loss': 1.0}, None, None)


class TestReduceLROnPlateau:
    def test_reduces_lr(self):
        opt = MockOptimizer(lr=0.01)
        scheduler = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, verbose=False
        )
        scheduler.on_train_begin()

        scheduler.on_epoch_end(0, {'val_loss': 1.0}, None, opt)
        scheduler.on_epoch_end(1, {'val_loss': 1.0}, None, opt)
        scheduler.on_epoch_end(2, {'val_loss': 1.0}, None, opt)

        new_lr = float(opt.learning_rate)
        assert new_lr < 0.01
