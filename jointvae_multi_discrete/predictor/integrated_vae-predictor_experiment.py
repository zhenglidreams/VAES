import sys

if '../' not in sys.path:
    sys.path.append('../')

import random
import csv
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
#from nas.AIS import Clonalg
from grammar_vae.NASGrammarModel import NASGrammarModel
from settings import settings as stgs
from grammar_vae.SentenceRetriever import SentenceRetrieverNoVaeTraining
from grammar_vae.nas_grammar import grammar
from integrated_predictor import IntegratedPredictor
from pytorch_model.Checkpoint import PredictorCheckpoint


n_pop = 2
min_depth = 3
max_depth = 9
window_pos = min_depth
max_window_size = 4
current_window_size = 1
tst_set = []
trn_set = []


def split_batch_file(dir_path, filename, test_pct = 0.2):
    full, tr, tst = [], [], []
    with open(dir_path/filename, 'r') as f:
        lines = f.readlines()
    for l in lines:  # ugly but it's only 2000 lines
        if l in tst_set or l in trn_set:
            if l in tst_set:
                tst.append(l)
            else:
                tr.append(l)
        else:
            if random.random() < test_pct:
                tst_set.append(l)
                tst.append(l)
            else:
                trn_set.append(l)
                tr.append(l)
    print(f'Split data into {len(tr)} training and {len(tst)} test samples.')
    with open(dir_path/'train.csv', 'w') as f:
        f.write(''.join(tr))
    with open(dir_path/'test.csv', 'w') as f:
        f.write(''.join(tst))


def split_population(path, filename, test_pct = 0.2):
    full_data = pd.read_csv(path/filename, names=['sentence', 'fitness'], index_col=False,
                                dtype={'sentences': str, 'fitness': np.float32})
    n_samples = full_data.shape[0]
    tr_idx = random.sample(list(range(n_samples)), int(n_samples * (1 - test_pct)))
    tst_idx = list(set(range(n_samples)).difference(tr_idx))
    print(f'Split data into {len(tr_idx)} training and {len(tst_idx)} test samples.')
    full_data.iloc[tr_idx, :].to_csv(stgs.PRED_BATCH_PATH/'train.csv')
    full_data.iloc[tst_idx, :].to_csv(stgs.PRED_BATCH_PATH/'test.csv')


def scrolling_gen_batch(dir_path, filename, win_size, win_pos, test_pct = 0.2):  # adds the batches in size window
    with open(dir_path/filename, 'r') as f:
        data = list(csv.reader(f))
    writer = csv.writer(
        open(dir_path/"windowbatch.csv", 'w', newline=''))
    node_sizes = list(range(win_pos, win_pos + win_size))
    print(f'Networks depths considered: {node_sizes}.')
    for row in data:
        cnt = row[0].count('/') - 1
        if cnt in node_sizes:
            writer.writerow(row)
    split_batch_file(stgs.PRED_BATCH_PATH, 'textfiles/windowbatch.csv', test_pct)


if __name__ == '__main__':
    import os, shutil
    from pathlib import Path
    import random

    def clear_files(path, file = None):
        # file == None: Remove everything. Otherwise only remove specified file.
        if isinstance(path, str):
            path = Path(path)
        if file is None:
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path/file)

    #stgs.LOG_PATH = Path('H:/gvae_exps/test_predictor/log')
    stgs.LOG_PATH = Path('test_predictor/log')
    #stgs.VAE_HPARAMS['weights_path'] = Path('H:/gvae_exps/test_predictor/vae_wts/weights_2048.pt')
    stgs.WTS_PATH = Path('test_predictor/vae_wts/weights_256.pt').as_posix()
    #stgs.PRED_BATCH_PATH = Path('H:/gvae_exps/test_predictor/pred_batches')
    stgs.PRED_BATCH_PATH = Path('test_predictor/pred_batches')
    #stgs.PRED_HPARAMS["weights_path"] = Path('H:/gvae_exps/test_predictor/integrated_pred_ckpts')
    stgs.PRED_HPARAMS["weights_path"] = Path('test_predictor/integrated_pred_ckpts')

    # delete previous logs and windowbatch.csv
    # clear_files(stgs.LOG_PATH)
    clear_files(stgs.PRED_BATCH_PATH/'textfiles/windowbatch.csv')

    #split_batch_file(stgs.PRED_BATCH_PATH, 'textfiles/fitnessbatch.csv')

    # Experiment 1:
    # # 1. Generate a random population of varying sizes
    # ais = Clonalg(n_pop, 1, 10, 0.5, 0.2, 1, 1, 1, 0.1, 0.5, False)
    # ais.clear_files(stgs.WTS_PATH)
    # ais.clear_files(stgs.PRED_BATCH_PATH/'textfiles')
    # ais.pop = make_population()
    # # print(ais.pop)
    # n_ntw = len(ais.pop)
    # print(f'{n_ntw} networks in population')
    # # # 2. Train it (for longer than the actual NAS procedure)
    # ais.evaluate(init=True)
    # # # 3. Split the trained population into a training set and a test set and write them to disk
    #  split_population(ais.pop, stgs.PRED_BATCH_PATH, 0.2)
    # 4. Create a grammar model
    g_mdl = NASGrammarModel(grammar, 'cuda')

    # 4. Train the predictor
    predictor = IntegratedPredictor(g_mdl, pretrained_vae=True, freeze_vae=False)
    logger = TensorBoardLogger(save_dir=stgs.LOG_PATH,
                               version='latent_sz_2048_trained_unfrozen_lr1e-4_0')
    it = 0
    ckpt_callback = PredictorCheckpoint(stgs.PRED_HPARAMS["weights_path"])
    # Initial trainer setup
    trainer = pl.Trainer(
        min_epochs=stgs.PRED_HPARAMS["max_epochs"],
        max_epochs=stgs.PRED_HPARAMS["max_epochs"],
        checkpoint_callback=False,  # we use our own checkpointing system
        callbacks=[ckpt_callback],
        val_percent_check=0.0,
        log_save_interval=1,
        weights_summary=None,
        gpus=1,
        logger=logger)

    while window_pos + current_window_size - 1 <= max_depth:
        print('NEW BATCH SCROLL Window Size = ' + str(current_window_size) + ', Window Position = ' + str(window_pos))
        scrolling_gen_batch(stgs.PRED_BATCH_PATH/'textfiles', 'fitnessbatch.csv', current_window_size,
                            window_pos, test_pct=0.20
                            )
        if current_window_size >= stgs.PRED_HPARAMS['warm_up_period']:  # let samples accumulate for a while
            trainer.fit(predictor)
            # 5. Evaluate the predictor's performance
            # predictor.load_from_checkpoint('H:/gvae_exps/test_predictor/pred_ckpts/default/version_68/checkpoints/epoch=1357.ckpt')
            trainer.test(predictor)
            predictor.eval()
            cols = {'truth': [], 'pred': [], 'absdiff': []}
            for batch in predictor.test_dataloader():
                one_hot, n_layers, y = batch
                one_hot, n_layers, y = one_hot.to('cuda'), n_layers.to('cuda'), y.to('cuda')

                y_hat = predictor((one_hot, n_layers))
                cols['truth'].append(y.item())
                cols['pred'].append(y_hat.item())
                cols['absdiff'].append((y_hat - y).abs().item())
            results = pd.DataFrame.from_dict(cols)
            results.to_csv('H:/gvae_exps/test_predictor/exp_integrated.csv')
            # results.to_csv('../../test_predictor/exp_1.csv')
            print(f"Spearman correlation: {results.corr('spearman').iloc[0,1]:.4f}")
            # prepare next iteration
            it += 1
            trainer = pl.Trainer(
                min_epochs = it * stgs.PRED_HPARAMS["max_epochs"],
                max_epochs = it * stgs.PRED_HPARAMS["max_epochs"],
                resume_from_checkpoint = stgs.PRED_HPARAMS["weights_path"]/"predictor.ckpt",
                checkpoint_callback = False,  # we use our own checkpointing system
                callbacks = [ckpt_callback],
                logger = logger,
                # monitor = 'batch_idx', save_top_k = 1, mode = 'max',
                val_percent_check = 0.0,
                log_save_interval = 1,
                gpus = 1,
                weights_summary = None,
            )

        if current_window_size < max_window_size:
            current_window_size = current_window_size + 1
        else:
            window_pos = window_pos + 1

