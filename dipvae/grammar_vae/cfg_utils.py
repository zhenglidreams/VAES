import sys

if '../' not in sys.path:
    sys.path.append('../')

import numpy as np
import nltk
import torch


debug = True
tokenize=None
def make_tokenizer(cfg: nltk.CFG):
    """
    Creates a tokenizer; temporarily replaces terminal symbols of length > 1 with a single character
    :param cfg: An nltk.CFG object
    :return: A tokenizer function.
    """
    long_tokens = list(filter(lambda a: len(a) > 1, cfg._lexical_index.keys()))
    replacements = ['£','$','%','^','&','*','#','@','?', '+', '¬', '¦'][:len(long_tokens)]
    # replacements = []
    assert (len(long_tokens)) == len(replacements)
    for token in replacements:  # make sure we are not replacing by a token that already exists
        assert token not in cfg._lexical_index.keys()

    global tokenize
    def tokenize(ntw_str):
        for i, token in enumerate(long_tokens):
            ntw_str = ntw_str.replace(token, replacements[i])
        tokens = []
        for token in ntw_str:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except:
                tokens.append(token)
        return tokens

    return tokenize

def make_prod_map(cfg: nltk.CFG):
    ''' Assigns an index to each production, in the form: {'<production>': <index>}. '''
    prod_map = {}
    for i, prod in enumerate(cfg.productions()):
        prod_map[prod] = i
    return prod_map

def make_one_hot(cfg: nltk.CFG, tokenizer, prod_map, sents, max_len = 25, n_chars = 34):
    """
    Encodes a list of sentences (strings) into a one-hot vector representing the production rules used to generate it.
    """
    if not isinstance(sents, list):
        sents = [sents]
    tokens = list(map(tokenizer, sents))  # tokenize sentences
    parse_trees = [next(nltk.ChartParser(cfg).parse(t)) for t in tokens]  # build parse tree for each sentence
    prod_seq = [tree.productions() for tree in parse_trees]  # list productions used in each parse tree
    indices = []  # list of vectors identifying the production rules used in each sentence
    for entry in prod_seq:
        indices.append(np.array([prod_map[prod] for prod in entry], dtype=int))
    one_hot = np.zeros((len(indices), max_len, n_chars), dtype=np.float32)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions), indices[i]] = 1.
        one_hot[i][np.arange(num_productions, max_len), -1] = 1.  # fill last column of
        # unused production slots with 1, which corresponds to the rule "Nothing -> None".
    return torch.tensor(one_hot)

def pr_sz(t, value = False):
    '''Utility function that displays the dimensions of a variable and, if required, its value. Only in debug mode.'''
    if debug:
        assert isinstance(t, str), 'Pass variable name as a string.'
        try:
            t_var = globals()[t]
        except:
            t_var = locals()[t]
        if isinstance(t_var, (torch.Tensor, np.ndarray)):
            print(t, ':', t_var.shape)
        elif isinstance(t_var, (list, dict, tuple, str)):
            print(t, ':', len(t_var))
        if value:
            print(t_var)
    else:
        pass

if __name__ == '__main__':
    from nas_grammar import grammar, cfg_str
    cfg = nltk.CFG.fromstring(cfg_str)

