from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import settings.settings as stgs


class LinearBNActiv(nn.Module):
    def __init__(self, dim_in, dim_out, batch_norm = True, activation = 'relu', droprate = 0.):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        self.droprate = droprate
        if activation == 'relu': self.layers.append(nn.ReLU(inplace=True))
        elif activation == 'tanh': self.layers.append(nn.Tanh())
        elif activation == 'gelu': self.layers.append(nn.GELU())
        elif activation == 'none': pass
        if batch_norm: self.layers.append(nn.BatchNorm1d(dim_out))
        if droprate > 0: self.layers.append(nn.Dropout(droprate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LinearBNActiv_for_MCDropout(nn.Module):
    def __init__(self, dim_in, dim_out, batch_norm = True, activation = 'relu', droprate = 0.):
        super().__init__()
        self.droprate = droprate
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        self.droprate = droprate
        if activation == 'relu': self.layers.append(nn.ReLU(inplace=True))
        elif activation == 'tanh': self.layers.append(nn.Tanh())
        elif activation == 'none': pass
        if batch_norm: self.layers.append(nn.BatchNorm1d(dim_out))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.dropout(x, p=self.droprate, training=True)
        return x


class MLPPredictor(nn.Module):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hp = hparams
        self.fc_dims = [int(s) for s in self.hp.hid_dims.split(",")]
        # add input and output dimensions:
        self.fc_dims = [self.hp.in_dim+self.hp.c_dim] + self.fc_dims
        # self.bn1 = nn.BatchNorm1d(self.fc_dims[0])
        if self.hp.mc_dropout:
            self.layers = nn.ModuleList(
                [LinearBNActiv_for_MCDropout(
                    self.fc_dims[i],
                    self.fc_dims[i + 1],
                    batch_norm = True,
                    activation = self.hp.activation,
                    droprate = self.hp.droprate_hidden)
                    for i in range(len(self.fc_dims) - 1)
                ])
        else:  # normal dropout, if any
            self.layers = nn.ModuleList(
                [LinearBNActiv(
                    self.fc_dims[i],
                    self.fc_dims[i + 1],
                    batch_norm=True,
                    activation = self.hp.activation,
                    droprate=self.hp.droprate_hidden)
                    for i in range(len(self.fc_dims) - 1)
                ])
            self.do = nn.Dropout(self.hp.droprate_head)
        self.last_layer = nn.Linear(self.fc_dims[-1], 1)
        self.bn2 = nn.BatchNorm1d(1)

        # weight initialisation
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # initialise final layer (found to initialize logits around 0.5 prob)
        self.last_layer.weight.data.fill_(0.)
        self.last_layer.weight.data += torch.randn_like(self.last_layer.weight.data) * 0.01

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for l in self.layers:
            x = l(x)
        x = self.bn2(self.last_layer(x))
        # apply dropout and custom activation function
        if self.hp.mc_dropout:  # with MC dropout, we want the same behaviour at training and inference
            x = F.dropout(x, p=self.hp.droprate_head, training=True)
        else:
            x = self.do(x)
        return 0.5 * (1 + torch.sin(self.do(x) / 4.0))

class DoubleLeakyReLUn(nn.Module):
    def __init__(self, alpha = 0.01, n = 6):
        super().__init__()
        self.n = n
        self.alpha = alpha

    def low_slope_bottom(self, x):
        return self.alpha * x

    def low_slope_up(self, x):
        return self.alpha * x + self.n * (1 - self.alpha)

    def forward(self, x):
        return torch.min(torch.max(self.low_slope_bottom(x), x), self.low_slope_up(x))

class ExU(nn.Module):
    """
    A single layer of exp-centered units (Agarwal et al. 2020: Neural Additive Models)
    """
    def __init__(self, dim_in, dim_out, batch_norm = True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = nn.parameter.Parameter(torch.Tensor(dim_out, dim_in))
        self.bias = nn.parameter.Parameter(torch.Tensor(dim_in))
        self.has_bn = batch_norm
        if self.has_bn: self.bn = nn.BatchNorm1d(dim_out)
        self.activation = DoubleLeakyReLUn(alpha=0.05, n=3)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, 4, 0.5)
        nn.init.uniform_(self.bias, -1, 1)

    def forward(self, x):
        out = (x - self.bias).matmul(torch.exp(self.weight).t())
        out = self.activation(out)
        if self.has_bn:
            out = self.bn(out)
        return out

class ExUPredictor(nn.Module):
    """
    Implements exp-centered units model (Agarwal et al. 2020: Neural Additive Models)
    """
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hp = hparams
        self.fc_dims = [int(s) for s in self.hp.hid_dims.split(",")]
        # add input and output dimensions:
        self.fc_dims = [self.hp.in_dim] + self.fc_dims
        self.layers = nn.ModuleList(
                [ExU(self.fc_dims[i], self.fc_dims[i + 1], batch_norm=True)
                    for i in range(len(self.fc_dims) - 1)
                ]
        )
        self.last_layer = ExU(self.fc_dims[-1], 1)
        self.bn = nn.BatchNorm1d(1)
        self.do = nn.Dropout(self.hp.drop_rate)
        self.out = DoubleLeakyReLUn(alpha = 0.01, n = 1)

        # weight initialisation
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # initialise final layer (found to initialize logits around 0.5 prob)
        self.last_layer.weight.data.fill_(1.)
        self.last_layer.weight.data += torch.randn_like(self.last_layer.weight.data) * 1e-4

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for l in self.layers:
            x = l(x)
        x = self.bn(self.last_layer(x))
        # apply dropout and double-leaky relu activation
        return self.out(self.do(x))

class EnsemblePredictor(nn.Module):
    """
    Implements an ensemble model with n inputs and a single output that is the average of n individual models' outputs.
    """
    def __init__(self, hparams: Namespace, n_models: int = 5):
        super().__init__()
        self.hp = hparams
        self.n_models = n_models
        self.models = nn.ModuleList([
            MLPPredictor(hparams) for _ in range(n_models)
        ])

    def forward(self, x):
        outputs = [m(x) for m in self.models]
        outputs = torch.stack(outputs, dim=1)
        return outputs.mean(dim=1)


