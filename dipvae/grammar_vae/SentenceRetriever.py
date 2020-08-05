import sys
if '../' not in sys.path:
    sys.path.append('../')

import random
from typing import Union, Optional, Tuple
from pathlib import Path
import nltk
from nltk.grammar import is_terminal, is_nonterminal
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from grammar_vae.NASGrammarModel import NASGrammarModel
from grammar_vae.cfg_utils import make_tokenizer, make_prod_map, make_one_hot
import settings.settings as stgs


class SentenceRetrieverNoVaeTraining(Dataset):
    """
    Take sentences and corresponding fitness as inputs (from a file), and returns latent vectors and
    fitness values. To achieves this, it loads a pretrained VAE and Grammar model and passes the sentence through the
    encoder, which generates a latent vector.
    """
    print(1)
    def __init__(self, file_pathname, grammar_model):
        super().__init__()
        self.data = pd.read_csv(file_pathname, names=['sentence', 'fitness'], index_col=False,
                                dtype={'sentences': str, 'fitness': np.float32})
        self.grammar_model = grammar_model
        
    
    def __getitem__(self, idx):
        sent, fitness = [self.data.iloc[idx, 0]], self.data.iloc[idx, 1]
        z, oh = self.grammar_model.encode(sent)
        n_layers = torch.tensor((sent.count('/') + 1) * 1.0 / stgs.PRED_HPARAMS['max_depth'])
        n_layers = n_layers.type_as(z).view(1, -1)
        z = torch.cat([z, n_layers], dim=1)
        return z, fitness
        

    def __len__(self):
        return self.data.shape[0]
        


class SentenceRetrieverWithVaeTraining(Dataset):
    """
    Take sentences and corresponding fitness as inputs (from a file), and returns a matrix of one-hot vectors
    (where each element corresponds to a step and a production rule), the number of layers of the network,
    and the fitness value of the network.
    """
    def __init__(self, file_pathname: Union[Path, str], grammar_mdl: NASGrammarModel):
        super().__init__()
        self.data = pd.read_csv(file_pathname, names=['sentence', 'fitness'], index_col=False,
                                dtype={'sentences': str, 'fitness': np.float32})
        self.gr_mdl = grammar_mdl

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        sent, fitness = [self.data.iloc[idx, 0]], self.data.iloc[idx, 1]
        one_hot = make_one_hot(
            self.gr_mdl._grammar.GCFG,
            self.gr_mdl._tokenize,
            self.gr_mdl._prod_map,
            sent,
            self.gr_mdl.max_len,
            self.gr_mdl._n_chars
            ).transpose(2, 1)
        n_layers = torch.tensor((sent.count('/') + 1) * 1.0 / stgs.PRED_HPARAMS['max_depth'],
                                dtype=torch.float32, requires_grad=False)
        return one_hot, n_layers, fitness

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    from pathlib import Path
    from torch.utils.data import DataLoader
    from grammar_vae.nas_grammar import grammar
    from grammar_vae.NASGrammarModel import NASGrammarModel

    stgs.PRED_BATCH_PATH = Path('../../runs/test_predictor/pred_batches')
    stgs.VAE_WTS_PATH = Path('../../runs/test_predictor/vae_wts')
    grammar_mdl = NASGrammarModel(grammar, 'cuda')
    dataset = SentenceRetrieverNoVaeTraining(stgs.PRED_BATCH_PATH / 'train.csv', grammar_mdl)
    dataloader = DataLoader(dataset,  batch_size=1)

    for batch in dataloader:
        x, y = batch
        print(x.shape)
        print(y)