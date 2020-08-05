import sys
if not '../' in sys.path:
    sys.path.append('../')
from argparse import Namespace
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import settings.settings as stgs
from grammar_vae.SentenceRetriever import SentenceRetrieverNoVaeTraining
from grammar_vae.nas_grammar import grammar
from grammar_vae.NASGrammarModel import NASGrammarModel
from models import MLPPredictor, ExUPredictor, EnsemblePredictor


class PerfPredictor(pl.LightningModule):
    def __init__(self, grammar_mdl, hparams=stgs.PRED_HPARAMS):
        super().__init__()
        self.hp = Namespace(**hparams)  # convert dict to argparse.Namespace
        self.grammar_mdl = grammar_mdl

        if self.hp.model_type == 'mlp':
            # use normal feed-forward model
            self.model = MLPPredictor(self.hp)
        elif self.hp.model_type == 'exu':
            # use exp-centered units  (Agarwal et al. 2020: Neural Additive Models)
            self.model = ExUPredictor(self.hp)
        elif self.hp.model_type == 'both':
            self.mlp = MLPPredictor(self.hp)
            self.exu = ExUPredictor(self.hp)
        elif self.hp.model_type == 'ensemble':
            self.model = EnsemblePredictor(self.hp, 3)

    def forward(self, x):
        if self.hp.model_type in ['mlp', 'exu', 'ensemble']:
            return self.model(x)
        elif self.hp.model_type == 'both':
            return self.mlp(x) + self.exu(x)

    def loss_function(self, y_pred, y_true):
        return F.mse_loss(y_pred.squeeze(1), y_true.squeeze())

    def configure_optimizers(self):
        self.optim = torch.optim.AdamW(
            self.parameters(), lr=self.hp.lr_ini, weight_decay=self.hp.w_decay
        )
        if self.hp.val_set_pct > 0.0:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim,
                "min",
                factor=self.hp.lr_reduce_on_plateau_factor,
                patience=self.hp.lr_reduce_patience,
                min_lr=self.hp.lr_min,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
                self.optim, lr_lambda=lambda epoch: self.hp.lr_reduce_mult_factor
            )
        return [self.optim], [self.scheduler]

    def training_step(self, batch, batch_idx):
        z, y_true = batch
        y_pred = self.forward(z)
        # print(y_pred)
        loss_val = self.loss_function(y_pred, y_true)
        lr = torch.tensor(self.optim.param_groups[0]["lr"]).type_as(loss_val)
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
        z, y_true = batch
        y_pred = self.forward(z)
        loss_val = self.loss_function(y_pred, y_true)
        return {'test_loss': loss_val.sqrt()}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}

    def prepare_data(self):
        try:
            tr_set = SentenceRetrieverNoVaeTraining(
                stgs.PRED_BATCH_PATH / "train.csv", grammar_model=self.grammar_mdl
            )
            self.tst_set = SentenceRetrieverNoVaeTraining(
                stgs.PRED_BATCH_PATH / "test.csv", grammar_model=self.grammar_mdl
            )
        except:
            tr_set = SentenceRetrieverNoVaeTraining(
                stgs.PRED_BATCH_PATH / "fitnessbatch.csv",
                grammar_model=self.grammar_mdl,
            )
        if self.hp.val_set_pct > 0.0:
            val_len = int(self.hp.val_set_pct * len(tr_set))
            self.tr_set, self.val_set = random_split(
                tr_set, [len(tr_set) - val_len, val_len]
            )
        else:
            self.tr_set, self.val_set = tr_set, None

    def train_dataloader(self):
        return DataLoader(
            self.tr_set,
            batch_size=self.hp.batch_sz,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(self.tst_set, batch_size=1, shuffle=False, drop_last=False)


if __name__ == "__main__":
    from pathlib import Path
    from pytorch_lightning.logging import TensorBoardLogger

    stgs.PRED_BATCH_PATH = Path("test_predictor/pred_batches")
    pred = PerfPredictor()
    logger = TensorBoardLogger("test_predictor/pred_ckpts")
    trainer = pl.Trainer(
        gpus=-1, early_stop_callback=None, max_epochs=stgs.PRED_HPARAMS["max_epochs"]
    )
    trainer.fit(pred)

    """
    pred = Predictor()
    incbatch = torch.rand((1, 128))
    incbactch = incbatch.view(1, 1, 128)
    output = pred(incbactch)
    print(output)
    optimiser = optim.Adam(pred.parameters(), lr=0.001)
    incbatch = torch.rand((10, 128))
    train = incbatch
    test = incbatch
    trainset = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)
    epochs = 3
    for epoch in range(epochs):
        for data in trainset:
            x = data
            pred.zero_grad()
            output = pred(x.view(1, 1, 128))
            loss = F.nll_loss(output, x)  # better for scalar inputs not one hot ?
            loss.backward()
            optimiser.step()
        print(loss)
        correct = 0
        total = 0
        with.torch.no_grad():  # Training/evaluation mode
            for data in testset():
                x = data
                output = pred(x.view(1, 1, 128))
                for idx, i in enumerate(output):
                    if torch.argmax(i) == x[idx]
                        correct += 1
                    total += 1
    """
