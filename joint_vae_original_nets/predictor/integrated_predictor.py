import sys
if not '../' in sys.path:
    sys.path.append('../')
from argparse import Namespace
from typing import Union, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
# from torch.utils.data.dataloader import _InfiniteConstantSampler
import pytorch_lightning as pl
import settings.settings as stgs
from grammar_vae.SentenceRetriever import SentenceRetrieverWithVaeTraining
from grammar_vae.nas_grammar import grammar
from grammar_vae.NASGrammarModel import NASGrammarModel
from grammar_vae.VAE import NA_VAE
from predictor import PerfPredictor


class IntegratedPredictor(pl.LightningModule):

    def __init__(
            self,
            grammar_mdl: NASGrammarModel,
            pretrained_vae: bool,
            freeze_vae: bool,
            vae_hparams: dict = stgs.VAE_HPARAMS,
            pred_hparams: dict = stgs.PRED_HPARAMS,
            ):
        super().__init__()
        # make separate Namespaces for convenience
        self.vae_hparams, self.pred_hparams = vae_hparams, pred_hparams

        # make Namespace of combined hyperparameters for compatibility with PL:
        self.hparams = {}
        for k, v in vae_hparams.items():
            self.hparams['_'.join(['vae', k])] = v
        for k, v in pred_hparams.items():
            self.hparams['_'.join(['pred', k])] = v
        self.hparams = Namespace(**self.hparams)
        self.vae = NA_VAE(self.vae_hparams)
        if pretrained_vae:
            self.vae.load_state_dict(torch.load(self.hparams.vae_weights_path))
        if freeze_vae:
            self.vae.freeze()
            print('VAE encoder frozen.')
        self.predictor = PerfPredictor(grammar_mdl, self.pred_hparams)

    def forward(self, batch) -> torch.Tensor:
        one_hot, n_layers = batch
        mu, logvar, q = self.vae.encode(one_hot.squeeze(1))
        z=self.vae.reparameterize(mu, logvar, q)
        # Add explicit information on depth of network
        
        return self.predictor(z)

    def mixup_inputs(self, batch, alpha=1.0):
        """
        Returns mixed pairs of inputs and targets within a batch, and a lambda value sampled from a beta
        distribution.
        """
        one_hot, n_layers, y_true = batch
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.

        idx = torch.randperm(y_true.size(0))
        mixed_onehot = lam * one_hot + (1 - lam) * one_hot[idx, ...]
        mixed_n_layers = lam * n_layers + (1 - lam) * n_layers[idx]
        return mixed_onehot, mixed_n_layers, y_true, y_true[idx], lam

    def mixup_criterion(self, criterion, y_pred, y_true_a, y_true_b, lam):
        return lam * criterion(y_pred, y_true_a) + (1 - lam) * criterion(y_pred, y_true_b)

    def loss_function(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.predictor.loss_function(y_pred, y_true)

    def configure_optimizers(self):
        return self.predictor.configure_optimizers()

    def training_step(self, batch, batch_idx):
        one_hot, n_layers, y_true = batch
        if one_hot.dim() < 3: one_hot.unsqueeze_(0)
        if n_layers.dim() < 2: n_layers.unsqueeze_(0)
        if y_true.dim() < 2: y_true.unsqueeze_(0)

        if self.hparams.pred_mixup:
            one_hot, n_layers, y_true_a, y_true_b, lam = self.mixup_inputs(
                batch,
                self.hparams.pred_mixup_alpha
            )

        y_pred = self.forward((one_hot, n_layers))

        if self.hparams.pred_mixup:
            loss_val = self.mixup_criterion(self.loss_function, y_pred, y_true_a, y_true_b, lam)
        else:
            loss_val = self.loss_function(y_pred, y_true)

        lr = torch.tensor(self.predictor.optim.param_groups[0]["lr"]).type_as(loss_val)
        step = torch.tensor(self.global_step).type_as(lr)
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val.unsqueeze_(0)
            lr.unsqueeze_(0)
            step.unsqueeze_(0)
        logs = {"loss": loss_val.sqrt(), "lr": lr, "step": step}
        p_bar = {"global_step": step}
        return {
            "loss": loss_val.sqrt(),
            "lr": lr,
            "log": logs,
            "global_step": step,
            "progress_bar": p_bar,
        }

    def test_step(self, batch, batch_idx):
        one_hot, n_layers, y_true = batch
        y_pred = self.forward((one_hot, n_layers))
        loss_val = self.loss_function(y_pred, y_true)
        return {'test_loss': loss_val.sqrt()}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}

    def prepare_data(self):
        try:
            tr_set = SentenceRetrieverWithVaeTraining(
                stgs.PRED_BATCH_PATH / "train.csv", grammar_mdl=self.predictor.grammar_mdl
            )
            self.tst_set = SentenceRetrieverWithVaeTraining(
                stgs.PRED_BATCH_PATH / "test.csv", grammar_mdl=self.predictor.grammar_mdl
            )
        except:
            tr_set = SentenceRetrieverWithVaeTraining(
                stgs.PRED_BATCH_PATH / "fitnessbatch.csv", grammar_mdl=self.predictor.grammar_mdl,
            )
        if self.hparams.pred_val_set_pct > 0.0:
            val_len = int(self.hp.val_set_pct * len(tr_set))
            self.tr_set, self.val_set = random_split(
                tr_set, [len(tr_set) - val_len, val_len]
            )
        else:
            self.tr_set, self.val_set = tr_set, None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.tr_set,
            batch_size=self.hparams.pred_batch_sz,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.pred_num_workers,
        )

    def test_dataloader(self):
        return DataLoader(self.tst_set, batch_size=1, shuffle=False, drop_last=False,
                          num_workers=self.hparams.pred_num_workers)


if __name__ == "__main__":
    from pathlib import Path
    from pytorch_lightning.logging import TensorBoardLogger

    grammar_mdl = NASGrammarModel(grammar, 'cuda')
    stgs.VAE_HPARAMS['weights_path'] = Path("test_predictor/vae_wts/weights_256.pt")
    stgs.PRED_BATCH_PATH = Path("test_predictor/pred_batches")
    pred = IntegratedPredictor(grammar_mdl, stgs.VAE_HPARAMS, stgs.PRED_HPARAMS)
    logger = TensorBoardLogger("test_predictor/integrated_pred_ckpts")
    trainer = pl.Trainer(
        gpus=-1, early_stop_callback=None, max_epochs=stgs.PRED_HPARAMS["max_epochs"]
    )
    trainer.fit(pred)