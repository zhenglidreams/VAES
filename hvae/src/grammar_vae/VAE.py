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





def compute_dimensions(input_sz, module):
    with torch.no_grad():
        x = torch.ones(1, *input_sz, dtype=torch.float)
        size = tuple(module(x).shape)
    return size


class joint_vae_encoder(nn.Module):
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

class joint_vae_decoder(nn.Module):
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
        self.latent_sz_1=self.hp['latent_1']
        self.latent_sz_2=self.hp['latent_2']
        self.debed_sz_1=self.hp['debed_sz_1']
        self.debed_sz_2=self.hp['debed_sz_2']
        self.bsz = self.hp['batch_size']
        # self.data_sz = self.hp['data_size']
        self.max_len = self.hp['max_len']
        self.grammar = grammar
        self.ind_of_ind = torch.tensor(grammar.ind_of_ind, device='cuda')
        self.masks = torch.tensor(grammar.masks, device='cuda')
        self.kld_weight = 0.
        self.iter = 0.
        self.hidden = self.hp['hidden']
        self.k_sizes=self.hp['k_sizes']
        self.strides = self.hp['strides']
        self.padding = self.hp['padding']
        self.hidden_d = self.hp['hidden_d']
        self.k_sizes_d = self.hp['k_sizes_d']
        self.strides_d = self.hp['strides_d']
        self.padding_d = self.hp['padding_d']
        self.outpadding_d = self.hp['outpadding_d']
        

        self.encoder_2 = joint_vae_encoder(self.n_chars, self.hidden, self.k_sizes, self.strides,self.padding)
        self.encoder_1 = joint_vae_encoder(self.n_chars+1, self.hidden, self.k_sizes, self.strides, self.padding)
        #_, self.H, self.W = compute_dimensions((self.n_chars, self.max_len), self.beta_encoder)

        self.decoder=joint_vae_decoder(self.n_chars, self.hidden_d, self.k_sizes_d, self.strides_d,
                                         self.padding_d,self.outpadding_d)

        self.mu_1 = nn.Linear(self.hidden[-1] * 4, self.latent_sz_1)
        self.logvar_1 = nn.Linear(self.hidden[-1] * 4, self.latent_sz_1)
        self.mu_2 = nn.Linear(self.hidden[-1] * 4, self.latent_sz_2)
        self.logvar_2 = nn.Linear(self.hidden[-1] * 4, self.latent_sz_2)
        self.embed_input = nn.ConvTranspose1d(self.n_chars, self.n_chars, kernel_size=(1), stride=(1))
        self.embed_2 = nn.Linear(self.latent_sz_2, self.max_len)
        self.debed_1 = nn.Linear(self.latent_sz_1, self.debed_sz_1)
        self.debed_2 = nn.Linear(self.latent_sz_2, self.debed_sz_2)
        self.recons_mu_1 = nn.Linear(self.latent_sz_2, self.latent_sz_1)
        self.recons_logvar_1 = nn.Linear(self.latent_sz_2, self.latent_sz_1)
        self.decode_input = nn.Linear(self.debed_sz_1 + self.debed_sz_1, self.hidden[-1] * 4)


    def encode(self, x):
        h_2=self.encoder_2(x)
        h_2=torch.flatten(h_2, start_dim=1)
        mu_2_=self.mu_2(h_2)
        logvar_2_ = self.logvar_2(h_2)
        z2 = self.reparameterize(mu_2_, logvar_2_)
        embed_x=self.embed_input(x)
        embed_z2=self.embed_2(z2)
        embed_z2 = embed_z2.view(-1, self.max_len).unsqueeze(1)
        encoder1_input = torch.cat([embed_x, embed_z2], dim=1)
        encoder1_output=self.encoder_1(encoder1_input)
        encoder1_output=torch.flatten(encoder1_output, start_dim=1)
        mu_1_ = self.mu_1(encoder1_output)
        logvar_1_ = self.logvar_1(encoder1_output)
        return mu_1_, logvar_1_, mu_2_,logvar_2_, z2

    def decode(self, z):
        h = self.decode_input(z)
        h = h.view(-1, self.hidden[-1], 4)
        decode_ouput = self.decoder(h)
        return decode_ouput

    def forward(self, x):
        mu_1, logvar_1, mu_2,logvar_2, z2 = self.encode(x.squeeze())
        z1 = self.reparameterize(mu_1, logvar_1)
        debed_z1 = self.debed_1(z1)
        debed_z2 = self.debed_2(z2)
        z=torch.cat([debed_z1, debed_z2], dim=1)
        x_recon = self.decode(z)
        return x_recon,x,mu_1,logvar_1,mu_2,logvar_2,z1,z2


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def conditional(self, x_true, x_pred):
        most_likely = torch.argmax(x_true, dim=1)
        lhs_indices = torch.index_select(self.ind_of_ind, 0, most_likely.view(-1)).long()
        M2 = torch.index_select(self.masks, 0, lhs_indices).float()
        M3 = M2.reshape((-1, self.max_len, self.n_chars))
        M4 = M3.permute((0, 2, 1))
        P2 = torch.exp(x_pred) * M4
        P2 = P2 / (P2.sum(dim=1, keepdim=True) + 1.e-10)
        return P2

    def loss_function(self, x_decoded_mean, x_true, mu_z1,logvar_z1,mu_z2,logvar_z2,z1,z2):
        x_cond = self.conditional(x_true.squeeze(), x_decoded_mean)
        x_true = x_true.view(-1, self.n_chars * self.max_len)
        x_cond = x_cond.view(-1, self.n_chars * self.max_len)
       
        assert x_cond.shape == x_true.shape
        bce = F.binary_cross_entropy(x_cond, x_true, reduction='sum')
        p_mu_1 = self.recons_mu_1(z2)
        p_logvar_1 = self.recons_logvar_1(z2)
        kl_z1 = torch.mean(-0.5 * torch.sum(1 + logvar_z1 - mu_z1 ** 2 - logvar_z1.exp(), dim=1),dim=0)
        kl_z2 = torch.mean(-0.5 * torch.sum(1 + logvar_z2 - mu_z2 ** 2 - logvar_z2.exp(), dim=1),dim=0)
        kl_p_z1 = torch.mean(-0.5 * torch.sum(1 + p_logvar_1 - (z1 - p_mu_1) ** 2 - p_logvar_1.exp(),dim=1), dim=0)
        
        kl_loss = -(kl_p_z1 - kl_z1 - kl_z2)
        loss = bce+ self.kld_weight * kl_loss
        return loss,bce,-kl_loss
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
        decode_output,x,mu_1,logvar_1,mu_2,logvar_2,z1,z2= self.forward(x)
        loss_val, bce, kl_div = self.loss_function(decode_output,x,mu_1,logvar_1,mu_2,logvar_2,z1,z2)
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
        decode_output,x,mu_1,logvar_1,mu_2,logvar_2,z1,z2 = self.forward(x)
        loss_val, _, _ = self.loss_function(decode_output,x,mu_1,logvar_1,mu_2,logvar_2,z1,z2)
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

