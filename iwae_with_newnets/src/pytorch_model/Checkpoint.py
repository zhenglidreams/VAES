import sys
if '../' not in sys.path:
    sys.path.append('../')
import os
import logging
import warnings
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from settings import settings


class ArchitectureCheckpoint(ModelCheckpoint):
    """
    Minor variation of Pytorch Lightning's ModelCheckpoint that overrides the default filename and save path
    implemented in the original. This allows us to save weight files with the network ID as filename.
    """
    def __init__(self, filepath, save_best = True):
        self.filepath = filepath
        if save_best:
            save_top_k = 1
        else:
            save_top_k = -1
        super().__init__(filepath, monitor='val_acc', verbose=True, save_top_k=save_top_k, save_weights_only=True,
                         mode='max', period=1)
        self.epochs_since_last_check = 0

    def on_validation_end(self, trainer, pl_module):
        self.pl_module = pl_module
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        self.epochs_since_last_check += 1

        if self.save_top_k == 0:
            # no models are saved
            return
        if self.epochs_since_last_check >= self.period:
            self.epochs_since_last_check = 0

            filepath = self.format_checkpoint_name(epoch, metrics)
            # version_cnt = 0
            # while os.path.isfile(filepath):
            #     filepath = self.format_checkpoint_name(epoch, metrics, ver=version_cnt)
            #     # this epoch called before
            #     version_cnt += 1

            if self.save_top_k != -1:
                current = metrics.get(self.monitor)

                if current is None:
                    warnings.warn(
                        f'Can save best model only with {self.monitor} available,'
                        ' skipping.', RuntimeWarning)
                else:
                    if self.check_monitor_top_k(current):
                        self._do_check_save(filepath, current, epoch)
                        with open(settings.RUN_PATH / 'metric.tmp', 'w') as f:
                            f.write(str(current.item()))

                    else:
                        if self.verbose > 0:
                            logging.info(
                                f'\nEpoch {epoch:05d}: {self.monitor}'
                                f' was not in top {self.save_top_k}')

            else:
                if self.verbose > 0:
                    logging.info(f'\nEpoch {epoch:05d}: saving model to {filepath}')
                self._save_model(filepath)
                current = metrics.get(self.monitor)
                with open(settings.RUN_PATH / 'metric.tmp', 'w') as f:
                    f.write(str(current.item()))

    def format_checkpoint_name(self, epoch, metrics, ver=None):
        return os.path.join(self.dirpath, self.pl_module.id)

    def _save_model(self, filepath):
        '''
        Overrides original method to prevent previous weight files from being overwritten.
        :param filepath:
        :return:
        '''
        # make paths
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # delegate the saving to the model
        torch.save(self.pl_module.state_dict(), filepath)


class PredictorCheckpoint(Callback):
    """
    Variation of Pytorch Lightning's ModelCheckpoint that overrides the default filename and save path
    implemented in the original, and saves the last epoch even if validation is not run.
    """
    def __init__(self, dirpath):
        self.dirpath = dirpath

    def on_epoch_end(self, trainer, pl_module):
        self.pl_module = pl_module
        self.trainer = trainer
        filepath = self.format_checkpoint_name()
        self._save_model(filepath)

    def format_checkpoint_name(self):
        return os.path.join(self.dirpath, 'predictor.ckpt')

    def _save_model(self, filepath):
        # make paths
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # delegate the saving to the model
        self.trainer.save_checkpoint(filepath)
