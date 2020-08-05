import sys
if '../' not in sys.path:
    sys.path.append('../')

import random
import nltk
import itertools
# from nltk.grammar import is_terminal, is_nonterminal
import numpy as np
import torch
from torch.utils.data import IterableDataset
from grammar_vae.cfg_utils import make_tokenizer, make_prod_map, make_one_hot
import settings.settings as stgs


class SentenceGenerator(IterableDataset):
    def __init__(self, cfg, min_sample_depth, max_sample_depth, batch_size=256, seed=0):
        """
        :param cfg: An nltk.CFG object
        :param min_sample_depth:
        :param max_sample_depth:
        :param batch_size:
        """
        super().__init__()
        random.seed(seed)
        self.cfg = cfg
        self.bsz = batch_size
        self.tokenizer = make_tokenizer(cfg)
        self.prod_map = make_prod_map(cfg)
        self.weighted_sampling = stgs.VAE_HPARAMS['weighted_sampling']
        self.temp = stgs.VAE_HPARAMS['temperature']
        self.min_sample_depth, self.max_sample_depth = min_sample_depth, max_sample_depth  # min and max lengths of the
                                                                                   # sequences to sample
        if self.weighted_sampling:
            len_range = max_sample_depth - min_sample_depth
            self.probs = np.array([np.exp(self.temp * n / len_range) for n in range(min_sample_depth, max_sample_depth + 1)])
            self.probs /= self.probs.sum()

        self.max_len = stgs.VAE_HPARAMS['max_len']  # max possible length of a sequence
        self.lay_symb = stgs.VAE_HPARAMS['layer_symbol']  # symbol representing a new layer
        self.n_chars = len(cfg.productions())
        print(f'Grammar with {self.n_chars} productions.')
        self.sents = []

    def __iter__(self):
        while True:
            yield self.generate_one_hots()

    def length_sampler(self):
        return np.random.choice(np.arange(self.min_sample_depth, self.max_sample_depth + 1), size=1,
                                p=self.probs).item()

    def generate_sentence(self):
        if self.weighted_sampling:
            sampled_len = self.length_sampler()
            max_sample_len = min_sample_len = sampled_len
        else:
            sampled_len = None
            min_sample_len, max_sample_len = self.min_sample_depth, self.max_sample_depth

        def rewrite_at(index, replacements, the_list):
            del the_list[index]
            the_list[index:index] = replacements

        def get_valid_prods(symbol, n_layers):
            S = self.cfg.productions(lhs=symbol)  # subset of all prods with symbol as their lhs
            if n_layers < min_sample_len:  # to keep adding layers, we must sample prods with LAY in RHS
                lay_in_rhs = [p for p in S if nltk.grammar.Nonterminal('LAY') in p.rhs()]
                nonterm_in_rhs = [p for p in S if [pp for pp in p.rhs() if nltk.grammar.is_nonterminal(pp)]]
                if lay_in_rhs:
                    valid = lay_in_rhs
                elif nonterm_in_rhs:
                    valid = nonterm_in_rhs
                else:
                    valid = S
                return valid

            if n_layers >= max_sample_len or n_layers >= self.max_len:  # to stop adding layers, we want terminals
                no_lay_in_rhs = [p for p in S if not nltk.grammar.Nonterminal('LAY') in p.rhs()]
                all_terminals_in_rhs = [p for p in S if not [pp for pp in p.rhs() if nltk.grammar.is_nonterminal(pp)]]
                if no_lay_in_rhs:
                    valid = no_lay_in_rhs
                elif all_terminals_in_rhs:
                    valid = all_terminals_in_rhs
                else:
                    valid = S
                return valid
            return S

        n_layers = 1  # input layer only at this point
        sent = [self.cfg.start()]
        all_terminals = False
        while not all_terminals:
            all_terminals = True
            for position, symbol in enumerate(sent):
                if symbol in self.cfg._lhs_index:
                    all_terminals = False
                    valid_derivations = get_valid_prods(symbol, n_layers)
                    # derivations = self.cfg._lhs_index[symbol]
                    derivation = random.choice(valid_derivations)
                    rewrite_at(position, derivation.rhs(), sent)
                n_layers = sent.count('/') + 1  # /!\ once '/F' is inserted into the list, this no longer is the exact
                # number of layers as defined by the NAS algorithm, because it includes the classfier layer. But this
                # does not prevent this algorithm from returning sentences with the correct number of layers.
        return ''.join(sent), 1.0 * n_layers / self.max_sample_depth 

    def generate_one_hots(self):
        generated = [self.generate_sentence() for _ in range(self.bsz)]
        self.sents = [g[0] for g in generated]
        lengths = [g[1] for g in generated]
        # self.sents = self.generate_sentence()
        out = make_one_hot(self.cfg, self.tokenizer, self.prod_map, self.sents, max_len=self.max_len,
                           n_chars=self.n_chars)
        return out.transpose(-2, -1), torch.tensor(lengths, dtype=torch.float32)

if __name__ == '__main__':
    # testing
    import numpy as np
    from torch.utils.data import DataLoader
    from nas_grammar import grammar

    gen = SentenceGenerator(grammar.GCFG, 3, 10, batch_size=10000)
    dataloader = DataLoader(gen, batch_size=1)

    # def repeater(dataloader):
    #     for loader in itertools.repeat(dataloader):
    #         for data in loader:
    #             yield data

    # dataloader = repeater(dataloader)

    batch = next(iter(dataloader))
    oh, lens = batch
    orig_one_hots = oh.to('cuda')

    nothings = orig_one_hots.sum(-1)[..., -1]
    # print(nothings)
    # print(gen.max_len)
    max_seq_len = gen.max_len - nothings.min().item()  # maximum length of sequences generated by the generator
    print(max_seq_len)

    # for batch in dataloader:
    #     print(batch.shape)
    #     oh = batch
    #     orig_one_hots = oh.to('cuda')
    #     orig_sents = gen.sents
    #     print(orig_sents)