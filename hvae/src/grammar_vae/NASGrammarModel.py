import sys

if '../' not in sys.path:
    sys.path.append('../')

import numpy as np
import nltk
import torch
import settings.settings as stgs
from grammar_vae.VAE import NA_VAE
from grammar_vae.cfg_utils import make_prod_map, make_tokenizer, make_one_hot
from grammar_vae.nas_grammar import grammar


class NASGrammarModel():
    def __init__(self, grammar, device, hparams=stgs.VAE_HPARAMS):
        """
        Load trained encoder/decoder and grammar model
        :param grammar: A nas_grammar.Grammar object
        :param hparams: dict, hyperparameters for the VAE and the grammar model
        """
        self._grammar = grammar
        self.device = device
        self.hp = hparams
        self.max_len = self.hp['max_len']
        self._productions = self._grammar.GCFG.productions()
        self._prod_map = make_prod_map(grammar.GCFG)
        self._parser = nltk.ChartParser(grammar.GCFG)
        self._tokenize = make_tokenizer(grammar.GCFG)
        self._n_chars = len(self._productions)
        self._lhs_map = grammar.lhs_map
        self.vae = NA_VAE(self.hp)
        self.vae.eval()

    @staticmethod
    def _pop_or_nothing(stack):
        # Tries to pop item at top of stack S, unless there is nothing.
        try: return stack.pop()
        except: return 'Nothing'

    @staticmethod
    def _prods_to_sent(prods):
        # converts a list of productions into a sentence
        seq = [prods[0].lhs()]
        for prod in prods:
            if str(prod.lhs()) == 'Nothing':
                break
            for i, s in enumerate(seq):
                if s == prod.lhs():
                    seq = seq[:i] + list(prod.rhs()) + seq[i + 1:]
                    break
        try:
            return ''.join(seq)
        except:
            return ''

    def encode(self, sents):
        """
        Returns the mean of the distribution, which is the predicted latent vector, for a one-hot vector of production
        rules.
        """
        one_hot = make_one_hot(self._grammar.GCFG, self._tokenize, self._prod_map, sents, self.max_len,
                               self._n_chars).transpose(2, 1)  # (1, batch, max_len, n_chars)
        one_hot = one_hot.to(self.device)
        self.vae.eval()
        with torch.no_grad():
          mu_1, logvar_1, mu_2, logvar_2, z2 = self.vae.encode(one_hot)
          z1 = self.vae.reparameterize(mu_1, logvar_1)
          debed_z1 = self.vae.debed_1(z1)
          debed_z2 = self.vae.debed_2(z2)
          z = torch.cat([debed_z1, debed_z2], dim=1)
          
        return z, one_hot  # (batch, latent_sz)

    def _sample_using_masks(self, unmasked, logs = True):
        """
        Samples a one-hot vector from unmasked selection, masking at each timestep.
        /!\ This is probably where we will diverge from the Grammar-VAE paper because we need to introduce
        conditions on the nodes that can be selected as input for each layer, and the Agg types also
        depend on the node values selected (Agg == '-' iff ND == '-').
        :param unmasked: The output of the VAE's decoder, so a collection of logit vectors (i.e. before
        softmax); size (batch, timesteps, max_length)
        """
        x_hat = np.zeros_like(unmasked)
        # Create a stack (data structure) for each input in the batch, i.e. each sentence
        S = np.empty((unmasked.shape[0],), dtype=object)  # dimension 0 == number of sentences
        for i in range(S.shape[0]):
            S[i] = [str(self._grammar.start_index)]  # initialise each stack with the start symbol 'S'
        # Loop over time axis, sampling values and updating masks at every step
        for t in range(unmasked.shape[2]):
            next_nonterminal = [self._lhs_map[self._pop_or_nothing(a)] for a in S]
            mask = self._grammar.masks[next_nonterminal]  # get indices of valid productions for next symbol
            if logs:
                masked_output = np.multiply(np.exp(unmasked[..., t]), mask) + 1e-100
            else:
                masked_output = np.multiply(unmasked[..., t], mask) + 1e-100
                # This comes from Kusner et al. 2016 - GANs for Sequences of Discrete Elements with the
                # Gumbel-Softmax Distribution, using work done in Jang et al. 2017 - Categorical
                # Reparameteterization with Gumbel-Softmax, which itself uses the Gumber-Max trick presented
                # in Maddison et al. 2014 - A* Sampling. y ~ Softmax(h) is equivalent to setting
                # y=one_hot(argmax((h_i + g_i))) where g_i are independently sampled from Gumbel(0, 1)
            sampled_output = np.argmax(np.add(np.random.gumbel(size=masked_output.shape),
                                              np.log(masked_output))
                                       , axis=-1)
            # Fill the highest-probability production rule with 1., all others are 0.
            x_hat[np.arange(unmasked.shape[0]), sampled_output, t] = 1.
            # Collect non-terminals in RHS of the selected production and push them onto stack in reverse order
            rhs = [filter(lambda a: (isinstance(a, nltk.grammar.Nonterminal) and (str(a) != 'None')),
                          self._productions[i].rhs()) for i in sampled_output]  # single output per sentence
            for i in range(S.shape[0]):
                S[i].extend(list(map(str, rhs[i]))[::-1])
            if not S.any(): break  # stop when stack is empty
        return x_hat

    def decode(self, z = None, one_hot = None):
        """
        Sample from the grammar decoder using the CFG-based mask, and return a sequence of production rules.
        :param z: latent vector representing the sentence, of dimensions (batch, latent_sz). If None, must provide
        argument one_hot (for testing purposes, mainly).
        :param one_hot: If provided, decode the one_hot matrix directly instead of the decoded latent vector.
        """
        if z is None:  # testing purposes
            unmasked = one_hot
            logs = False
        else:  # normal regime
            logs = True
            #assert z.ndim == 2  # (batch, latent_sz)
            self.vae.eval()
            with torch.no_grad():
                unmasked = self.vae.decode(z).detach().cpu().numpy()  # (batch, max_len, n_char)
        assert unmasked.shape[1:] == (self._n_chars, self.max_len), \
            print(f'umasked_shape[1:] == {unmasked.shape[1:]}, expected {(self._n_chars, self.max_len)}.')
        x_hat = self._sample_using_masks(unmasked, logs)
        # Convert from one-hot to sequence of production rules
        prod_seq = [[self._productions[x_hat[index, :, t].argmax()]
                     for t in range(x_hat.shape[2])]
                    for index in range(x_hat.shape[0])]
        return [self._prods_to_sent(prods) for prods in prod_seq], x_hat


def compute_recon_acc(recon_one_hots, target_one_hots):
    print(recon_one_hots.shape, target_one_hots.shape)
    timestep_eq = np.all(recon_one_hots == target_one_hots, axis=1)
    sample_eq = timestep_eq.sum()
    return 1.0 * sample_eq.sum() / (target_one_hots.shape[0] * target_one_hots.shape[2])


if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    from time import time
    from datetime import datetime
    from SentenceGenerator import SentenceGenerator
    from scipy import stats
    import pytorch_lightning as pl
    from pytorch_lightning.logging import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    # from test_tube import Experiment

    value = True
    debug = True
    cwd = os.getcwd()
    seed = int(time()); print(f'Random Seed: {seed}')
    checkpoint_path = f'../predictor/test_predictor/vae_wts'

    def pr_sz(t, value = False):
        '''Utility function that displays the dimensions of a variable and, if required, its value. Only in debug mode.'''
        if debug:
            assert isinstance(t, str), 'Pass variable name as a string.'
            try:
                t_var = locals()[t]
            except:
                t_var = globals()[t]
            if isinstance(t_var, (torch.Tensor, np.ndarray)):
                print(t, ':', t_var.shape)
            elif isinstance(t_var, (list, dict, tuple, str)):
                print(t, ':', len(t_var))
            if value:
                print(t_var)
        else:
            pass

    min_depth = 3
    max_depth = stgs.VAE_HPARAMS['max_depth']  # maximum network depth

    print(f'Using maximum sequence length of {stgs.VAE_HPARAMS["max_len"]}.')
    torch.cuda.empty_cache()
    vae = NA_VAE(stgs.VAE_HPARAMS)
    vae = vae.float()
    vae=vae.cuda()
    # vae.load_state_dict(torch.load(f'{checkpoint_path}/weights.pt'))

    torch.cuda.empty_cache()
    version = datetime.strftime(datetime.fromtimestamp(seed), '%Y-%m-%d..%H.%M.%S')
    logger = TensorBoardLogger(checkpoint_path, version=version)
    checkpoint = ModelCheckpoint(
        filepath = checkpoint_path,
        save_top_k=1,
        verbose = True,
        monitor = 'loss',
        mode = 'min')
    early_stop = EarlyStopping(
        monitor = 'loss',
        patience = stgs.VAE_HPARAMS['early_stop_patience'],
        verbose=True,
        mode='min'
    )
    max_steps = stgs.VAE_HPARAMS['max_steps']
    # kld loss annealing also depends on max length (or max epochs?)
    vae.get_data_generator(max(min_depth, max_depth - 1),
                           max_depth, seed=seed)

    trainer = pl.Trainer(gpus=-1,
                         val_check_interval=9999,
                         early_stop_callback=None,
                         distributed_backend=None,
                         logger=logger,
                         max_steps=max_steps,
                         max_epochs=max_steps,
                         checkpoint_callback=checkpoint,
                         weights_save_path=checkpoint_path)
    
    trainer.fit(vae)
    torch.save(vae.state_dict(), f'{checkpoint_path}/weights_256.pt')
    vae.get_data_generator(min_depth, max_depth, seed=int(time()))
    # # checkpoint = torch.load(cwd+'/base_ckpt/full_grammar.hdf5', map_location = lambda storage, loc: storage)
    # # vae.load_state_dict(checkpoint['state_dict'])
    vae.load_state_dict(torch.load(f'{checkpoint_path}/weights_256.pt'))
    vae=vae.cuda()
    vae.eval()
    
    grammar_mdl = NASGrammarModel(grammar, device='cuda')
    grammar_mdl.vae = vae
    orig_sents = []
    torch.cuda.empty_cache()
    gen = SentenceGenerator(grammar.GCFG, min_depth, max_depth, batch_size=200)
    dataloader = DataLoader(gen, batch_size=1)

    batch = next(iter(dataloader))
    orig_one_hots, lens = batch
    print(orig_one_hots.size())
    orig_one_hots, lens = orig_one_hots.to('cuda'), lens.to('cuda')
    orig_sents.extend(gen.sents)
    z, one_hots = grammar_mdl.encode(orig_sents)
    print(z.size())
    s_hat, recon_one_hots = grammar_mdl.decode(z=z)  

    pr_sz('recon_one_hots', True)
    pr_sz('orig_sents', True)
    pr_sz('s_hat', True)

    corrects = [p == q for p, q in zip(orig_sents, s_hat)]
    print(f'Proportion of correctly recovered sentences: {100. * sum(corrects) / len(corrects):.2f}%')

    print(f'Proportion of correctly recovered production rules:'
          f' {100 * compute_recon_acc(recon_one_hots, one_hots.cpu().numpy()):.2f}%')

    
    print('end processing')
    


