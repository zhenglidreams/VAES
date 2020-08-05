import sys

if '../' not in sys.path:
    sys.path.append('../')

from operator import mul
from functools import reduce
from collections import OrderedDict
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.utils.data.dataloader import _InfiniteConstantSampler
import pytorch_lightning as pl
import settings.settings as stgs
from grammar_vae.SentenceGenerator import SentenceGenerator
from grammar_vae.nas_grammar import grammar
import numpy as np





from IPython import embed

class Unflatten(nn.Module):
    def __init__(self, size):
        """
        :param size: tuple (channels, length)
        """
        super().__init__()
        self.h, self.w = size

    def forward(self, x):
        """
        :param x: 2d-Tensor of dimensions (n_batches, self.h * self.w)
        """
        n_batches = x.size(0)
        assert self.h * self.w == x.shape[-1]
        return x.view(n_batches, self.h, self.w)


def compute_dimensions(input_sz, module):
    with torch.no_grad():
        x = torch.ones(1, *input_sz, dtype=torch.float)
        size = tuple(module(x).shape)
    return size


class vae_encoder(nn.Module):
    def __init__(self, ch_in, hids, ks, ss, ps):
        super().__init__()
        en_modules=[]
        for i in range(len(hids)):
            en_modules.append(
                nn.Sequential(
                nn.Conv1d(ch_in, out_channels=hids[i],
                          kernel_size=ks[i], stride=ss[i], padding=ps[i]),
                nn.BatchNorm1d(hids[i]),
                nn.ReLU()))
            ch_in=hids[i]
        self.layers=nn.Sequential(*en_modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class vae_decoder(nn.Module):
    def __init__(self, ch_in, hids, ks, ss, ps,ops):
        super().__init__()
        modules = []
        for i in range(len(hids) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hids[i],
                                       hids[i + 1],
                                       kernel_size=ks[i],
                                       stride=ss[i],
                                       padding=ps[i],
                                       output_padding=ops[i]
                                       ),
                    nn.BatchNorm1d(hids[i + 1]),
                    nn.ReLU())
            )

        modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hids[-1],
                                       hids[-1],
                                       kernel_size=ks[-1],
                                       stride=ss[-1],
                                       padding=ps[-1],

                                       output_padding=ops[-1]
                                       ),
                    nn.BatchNorm1d(hids[-1]),
                    nn.ReLU(),
                    nn.Conv1d(hids[-1], ch_in,
                              kernel_size=2, padding=0),
                    nn.Tanh())
        )
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class NA_VAE(pl.LightningModule):

    def __init__(self, hparams=stgs.VAE_HPARAMS):

        super().__init__()
        self.hparams = Namespace(**hparams)
        self.hp = hparams  # alias
        # self.device = self.hp['device']
        self.n_chars = self.hp['n_chars']
        self.bsz = self.hp['batch_size']
        # self.data_sz = self.hp['data_size']
        self.max_len = self.hp['max_len']
        self.fc_dim = self.hp['fc_dim']
        self.latent_sz = self.hp['latent_size']
        self.rnn_hidden = self.hp['rnn_hidden']
        self.drop_rate = self.hp['drop_rate']
        self.data_gen = None
        self.grammar = grammar
        self.ind_of_ind = torch.tensor(grammar.ind_of_ind, device='cuda')
        self.masks = torch.tensor(grammar.masks, device='cuda')
        self.kld_weight = 0.
        self.iter = 0.
        self.hidden=self.hp['hidden']
        self.k_sizes=self.hp['k_sizes']
        self.strides=self.hp['strides']
        self.padding=self.hp['padding']
        self.hidden_d = self.hp['hidden_d']
        self.k_sizes_d = self.hp['k_sizes_d']
        self.strides_d = self.hp['strides_d']
        self.padding_d = self.hp['padding_d']
        self.outpadding_d = self.hp['outpadding_d']
        self.num_samples = self.hp['num_samples']
        self.encoder = vae_encoder(self.n_chars, self.hidden, self.k_sizes, self.strides,
                                         self.padding)
        #_, self.H, self.W = compute_dimensions((self.n_chars, self.max_len), self.beta_encoder)

        self.decoder=vae_decoder(self.n_chars, self.hidden_d, self.k_sizes_d, self.strides_d,
                                         self.padding_d,self.outpadding_d)

        self.mu = nn.Linear(self.hidden[-1] * 4, self.latent_sz)
        self.logvar = nn.Linear(self.hidden[-1] * 4, self.latent_sz)
        self.decode_input=nn.Linear(self.latent_sz,self.hidden[-1]*4)

    def add_latent_vectors(self, hot_vector):
        if self.hot_list_started == False:
            self.hot_list[0] = hot_vector
            self.hot_list_started = True
        else:
            self.hot_list.append(hot_vector)

    def send_latent_vectors(self):
        #  self.latent_vector = torch.cat((self.latent_vector, hot_vector), 1)  # appends hot vector to the batch
        self.latent_vector = torch.stack(self.hot_list, 0)  # appends hot vector to the batch
        self.hot_list_started = False
        return self.latent_vector  # Sends the latent vectors to the predictor

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        mu_ = self.mu(h)
        logvar_ = self.logvar(h)
        return mu_, logvar_

    def decode(self, z):
        B = z.size()[0]
        z = z.reshape(-1, self.latent_sz)
        h=self.decode_input(z)
        h=h.view(-1,self.hidden[-1],4)
        decode_ouput=self.decoder(h)
        return decode_ouput,B

    def forward(self, x):
        mu, logvar = self.encode(x.squeeze())
        mu = mu.repeat(self.num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        logvar = logvar.repeat(self.num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = self.reparameterize(mu, logvar)
        x_recon,B = self.decode(z)
        return x_recon, mu, logvar,B
        #是decode后  还原 原来值的结果 当然 跟原来值会有差距

    def reparameterize(self, mu, logvar):
        if self.training:
            eps_std = self.hp['epsilon_std']
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def conditional(self, x_true, x_pred):
        most_likely = torch.argmax(x_true, dim=1)
        lhs_indices = torch.index_select(self.ind_of_ind, 0, most_likely.view(-1)).long()
        M2 = torch.index_select(self.masks, 0, lhs_indices).float()
        M3 = M2.reshape((-1, self.max_len, self.n_chars))
        M4 = M3.permute((0, 2, 1))
        P2 = torch.exp(x_pred) * M4
        # P2 = torch.exp(x_pred)
        P2 = P2 / (P2.sum(dim=1, keepdim=True) + 1.e-10)
        return P2

    def loss_function(self, x_decoded_mean, x_true, mu, logvar,B):
        x_true = x_true.squeeze().repeat(self.num_samples, 1, 1)
        x_cond = self.conditional(x_true.squeeze(), x_decoded_mean)
        assert x_cond.shape == x_true.shape
        x_cond = x_cond.view([B, -1, self.n_chars, self.max_len])
        x_true = x_true.view([B, -1, self.n_chars, self.max_len])
        log_p_x_z = ((x_cond - x_true) ** 2).flatten(2).mean(-1)  # Reconstruction Loss [B x S]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=2)  ## [B x S]
        log_weight = (log_p_x_z + self.kld_weight * kl_loss)  # .detach().data
        weight = F.softmax(log_weight, dim=-1)
        loss = torch.mean(torch.sum(weight * log_weight, dim=-1), dim=0)

        return loss, log_p_x_z, kl_loss

    @staticmethod
    def check_sz(x, target_size):
        assert x.shape[1:] == target_size, f'Wrong spatial dimensions: {tuple(x.shape[1:])}; ' \
            f'expected {target_size}'

    def configure_optimizers(self):
        self.optim = torch.optim.Adam(self.parameters(), lr=self.hp['lr_ini'], eps=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min',
                                                                    factor=self.hp['lr_reduce_factor'],
                                                                    patience=self.hp['lr_reduce_patience'],
                                                                    min_lr=self.hp['lr_min'])
        return [self.optim], [self.scheduler]

    def training_step(self, batch, batch_idx):
        # Set up KL Divergence annealing
        t0, t1 = self.hp['kld_wt_zero_for_epochs'], self.hp['kld_wt_one_at_epoch']  # for kld annealing
        if self.iter < self.hp['kld_wt_zero_for_epochs']:
            self.kld_weight = self.hp['kld_zero_value']
        else:
            self.kld_weight = min(self.hp['kld_full_value'] * (self.iter - t0) / (t1 - t0),
                                  self.hp['kld_full_value'])

        x, lengths = batch
        x_recon, mu, logvar, B = self.forward(x)
        loss_val, bce, kl_div = self.loss_function(x_recon, x, mu, logvar, B)
        lr = torch.tensor(self.optim.param_groups[0]['lr'])
        progress_bar = OrderedDict({'bce': bce.mean(), 'kl_div': kl_div.mean(),
                                    'lr': lr,
                                    'iter': self.iter,
                                    'kld_weight': self.kld_weight})
        log = OrderedDict({'loss': loss_val.mean(dim=0), 'bce': bce.mean(dim=0), 'kl_div': kl_div.mean(dim=0),
                           'lr': lr,
                           'kld_weight':
            self.kld_weight})
        self.iter += 1
        return {'loss': loss_val,
                'bce': bce,
                'kl_div': kl_div,
                'kld_weight': self.kld_weight,
                'iter': self.iter,
                'lr': lr,
                'log': log,
                'progress_bar': progress_bar}


    def test_step(self, batch, batch_idx):
        if self.on_gpu:
            x = batch.cuda()
        else:
            x = batch
        x = batch
        recon_x, mu, logvar = self.forward(x)
        loss_val, _, _ = self.loss_function(recon_x, x, mu, logvar,batch_idx)
        return {'test_loss': loss_val.mean(dim=0)}

    def training_end(self, outputs):
        if not isinstance(outputs, list):
            outputs = [outputs]
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        bce_mean = torch.stack([x['bce'] for x in outputs]).mean()
        kld_mean = torch.stack([x['kl_div'] for x in outputs]).mean()
        kld_weight = outputs[-1]['kld_weight']
        lr = outputs[-1]['lr']
        progress_bar = OrderedDict({'bce_mean': bce_mean, 'kld_mean': kld_mean,
                                    'kld_weight': kld_weight, 'lr': lr,
                                    'iter': self.iter})
        log = OrderedDict({'loss': loss_mean, 'bce_mean': bce_mean, 'kld_mean': kld_mean,
                                    'kld_weight': kld_weight, 'lr': lr})
        return {'loss': loss_mean,
                'val_loss': loss_mean,
                'log': log,
                'progress_bar': progress_bar
                }

    def test_epoch_end(self, batch, batch_idx):
        if not isinstance(outputs, list):
            outputs = [outputs]
        loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        log = OrderedDict({'test_loss': loss_mean})
        return {'test_loss': loss_mean, 'log': log}

    def get_data_generator(self, min_sample_len, max_sample_len, seed=0):
        self.data_gen = SentenceGenerator(grammar.GCFG, min_sample_len, max_sample_len, self.bsz, seed)

    def train_dataloader(self):
        self.ldr = DataLoader(self.data_gen, batch_size=1, num_workers=0)
        return self.ldr

    def test_dataloader(self):
        return DataLoader(self.data_gen, batch_size=1, num_workers=0)


