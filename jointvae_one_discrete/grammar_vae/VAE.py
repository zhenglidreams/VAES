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
import pytorch_lightning as pl
import settings.settings as stgs
from grammar_vae.SentenceGenerator import SentenceGenerator
from grammar_vae.nas_grammar import grammar
import numpy as np

def compute_dimensions(input_sz, module):
    with torch.no_grad():
        x = torch.ones(1, *input_sz, dtype=torch.float32)
        size = tuple(module(x).shape)
    return size


class ConvOrDeconv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, deconv = False):
        super().__init__()
        conv_args = dict(in_channels = ch_in,
                         out_channels = ch_out,
                         kernel_size = kernel_size,
                         stride = stride,
                         padding = kernel_size // 2,
                         bias=False)
        self.conv = nn.Conv1d(**conv_args) if not deconv else nn.ConvTranspose1d(**conv_args,
                                                                                 output_padding=stride // 2)
        self.bn = nn.BatchNorm1d(ch_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvOrDeconvBlock(nn.Module):
    def __init__(self, ch_in, ch_out_list, ks_list, s_list, deconv = False):
        super().__init__()
        depth = len(ch_out_list)
        ch_in_list = [ch_in] + ch_out_list[:-1]
        assert len(ch_in_list) == len(ch_out_list) == len(ks_list) == len(s_list)
        self.layers = nn.ModuleList([ConvOrDeconv(cin, cout, ks, s, deconv) for cin, cout, ks, s in zip(ch_in_list,
                                                                                                        ch_out_list,
                                                                                                        ks_list,
                                                                                                        s_list)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NA_VAE(pl.LightningModule):

    def __init__(self, hparams=stgs.VAE_HPARAMS):
        super().__init__()

        self.hparams = Namespace(**hparams)

        self.hp = hparams  # alias
        self.n_chars = self.hp['n_chars']
        self.bsz = self.hp['batch_size']
        self.max_len = self.hp['max_len']
        self.conv_channels = [int(s) for s in self.hp['channels'].split(',')]
        self.conv_k_sizes = [int(s) for s in self.hp['k_sizes'].split(',')]
        self.conv_strides = [int(s) for s in self.hp['strides'].split(',')]
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
        self.alpha = self.hp["alpha"]
        self.categorical=self.hp['categorical_channel']
        self.min_temp = self.hp["j_temperature"]
        self.temp = self.hp["j_temperature"]
        self.cont_min = self.hp['con_min_capacity']
        self.cont_max = self.hp['con_max_capacity']
        self.disc_min = self.hp['dis_min_capacity']
        self.disc_max = self.hp['dis_max_capacity']
        self.cont_gamma = self.hp['cont_gamma']
        self.disc_gamma = self.hp['disc_gamma']
        self.cont_iter = self.hp['cont_iter']
        self.disc_iter = self.hp['disc_iter']
        self.anneal_rate = self.hp['anneal_rate']
        self.anneal_interval = self.hp['anneal_interval']



        self.cnn_encoder = ConvOrDeconvBlock(self.n_chars, self.conv_channels, self.conv_k_sizes, self.conv_strides,
                                         deconv=False)
        _, self.H, self.W = compute_dimensions((self.n_chars, self.max_len), self.cnn_encoder)


        self.encoder_fc = nn.Linear(self.H * self.W, self.fc_dim)
        self.encoder_do = nn.Dropout(self.drop_rate)
        self.mu = nn.Linear(self.fc_dim, self.latent_sz)  # keep 1 position free for length encoding
        self.logvar = nn.Linear(self.fc_dim, self.latent_sz)  # keep 1 position free for length encoding
        self.q_fc = nn.Linear(self.fc_dim, self.categorical)
        self.dec_rnn_fc_1 = nn.Linear(self.latent_sz+self.categorical, self.latent_sz)
        self.dec_rnn_do = nn.Dropout(self.drop_rate)
        self.dec_gru = nn.GRU(self.latent_sz, self.rnn_hidden, num_layers=4, batch_first=True)
        self.dec_rnn_fc_2 = nn.Linear(self.rnn_hidden, self.n_chars)
        self.dec_cnn_fc_1 = nn.Linear(self.latent_sz+self.categorical, self.H * self.W)
        self.dec_cnn_do = nn.Dropout(self.drop_rate)
        self.decoder = ConvOrDeconvBlock(self.conv_channels[-1],
                                         list(reversed(self.conv_channels[:-1])) + [self.n_chars],
                                         list(reversed(self.conv_k_sizes)), list(reversed(self.conv_strides)),
                                         deconv=True)


    def encode(self, x):
        h = self.cnn_encoder(x)
        h = torch.relu(self.encoder_do(self.encoder_fc(h.view(x.size(0), -1))))
        mu_ = self.mu(h)
        logvar_ = self.logvar(h)
        q_=self.q_fc(h)
        return mu_, logvar_,q_

    def decode(self, z):
        rnn_h = self.dec_rnn_do(self.dec_rnn_fc_1(z))
        rnn_h = rnn_h.unsqueeze(1)
        rnn_h = rnn_h.repeat((1, self.max_len, 1)); 
        self.check_sz(rnn_h, (self.max_len, self.latent_sz))
        out_rnn, _ = self.dec_gru(rnn_h); 
        self.check_sz(out_rnn, (self.max_len, self.rnn_hidden))
        out_rnn = self.dec_rnn_fc_2(out_rnn).transpose(-2, -1); 
        self.check_sz(out_rnn, (self.n_chars, self.max_len))
        h_cnn = torch.relu(self.dec_cnn_fc_1(z))
        h_cnn = self.dec_cnn_do(h_cnn)
        h_cnn = torch.relu(h_cnn).view(z.size(0), self.H, self.W)
        out_cnn = self.decoder(h_cnn); 
        self.check_sz(out_cnn, (self.n_chars, self.max_len))
        return out_rnn + out_cnn


    def forward(self, x):
        mu, logvar,q = self.encode(x.squeeze())
        z = self.reparameterize(mu, logvar,q)
        x_recon = self.decode(z)
        return x_recon, mu, logvar,q

    def reparameterize(self, mu, logvar,q):
        std = torch.exp(0.5 * logvar)
        e = torch.randn_like(std)
        z_new = e * std + mu
        u = torch.rand_like(q)
        eps = 1e-7
        g = - torch.log(- torch.log(u + eps) + eps)
        # Gumbel-Softmax sample
        s = F.softmax((q + g) / self.temp, dim=-1)
        s = s.view(-1, self.categorical)
        return torch.cat([z_new, s], dim=1)

    def conditional(self, x_true, x_pred):
        most_likely = torch.argmax(x_true, dim=1)
        lhs_indices = torch.index_select(self.ind_of_ind, torch.tensor(0), most_likely.view(-1)).long()
        M2 = torch.index_select(self.masks, torch.tensor(0), lhs_indices).float()
        M3 = M2.reshape((-1, self.max_len, self.n_chars))
        M4 = M3.permute((0, 2, 1))
        P2 = torch.exp(x_pred) * M4
        # P2 = torch.exp(x_pred)
        P2 = P2 / (P2.sum(dim=1, keepdim=True) + 1.e-10)
        return P2

    def loss_function(self, x_decoded_mean, x_true, mu, logvar,q,batch_id_x):
        x_cond = self.conditional(x_true.squeeze(), x_decoded_mean)
        x_true = x_true.view(-1, self.n_chars * self.max_len)
        x_cond = x_cond.view(-1, self.n_chars * self.max_len)
        
        assert x_cond.shape == x_true.shape
        bce = F.binary_cross_entropy(x_cond, x_true, reduction='sum')
        if batch_id_x % self.anneal_interval == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(- self.anneal_rate * batch_id_x),
                                   self.min_temp)
        
        q_p = F.softmax(q, dim=-1)
        disc_curr = (self.disc_max - self.disc_min) * self.iter / float(self.disc_iter) + self.disc_min
        disc_curr = min(disc_curr, np.log(self.categorical))
        eps = 1e-7
        h1 = q_p * torch.log(q_p + eps)
        h2 = q_p * np.log(1. / self.categorical + eps)
        kl_disc_loss = torch.mean(torch.sum(h1 - h2, dim=1), dim=0)
        cont_curr = (self.cont_max - self.cont_min) * self.iter/ float(self.cont_iter) + self.cont_min
        cont_curr = min(cont_curr, self.cont_max)
        kl_div = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        capacity_loss = self.disc_gamma * torch.abs(disc_curr - kl_disc_loss) + \
                        self.cont_gamma * torch.abs(cont_curr - kl_div)


        return self.alpha * bce + self.kld_weight * capacity_loss, bce, capacity_loss

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
        recon_x, mu, logvar, q = self.forward(x)
        loss_val, bce, kl_div = self.loss_function(recon_x, x, mu, logvar, q, batch_idx)
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
        loss_val, _, _ = self.loss_function(recon_x, x, mu, logvar)
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

