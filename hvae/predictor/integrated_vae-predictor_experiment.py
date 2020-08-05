import sys

if '../' not in sys.path:
    sys.path.append('../')

from random import shuffle
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
from pathlib import Path

n_pop = 2
min_depth = 3
max_depth = 9
window_pos = min_depth
max_window_size = 4
current_window_size = 1
tst_set = []
trn_set = []


def split_population(path, filename, test_pct = 0.2):
    full_data = pd.read_csv(os.path.join(path,filename), names=['sentence', 'fitness'], index_col=False,
                                dtype={'sentences': str, 'fitness': np.float32})
    n_samples = full_data.shape[0]
    tr_idx = random.sample(list(range(n_samples)), int(n_samples * (1 - test_pct)))
    tst_idx = list(set(range(n_samples)).difference(tr_idx))
    print(f'Split data into {len(tr_idx)} training and {len(tst_idx)} test samples.')
    full_data.iloc[tr_idx, :].to_csv(os.path.join(stgs.PRED_BATCH_PATH,'textfiles/full_train.csv'),index=False)
    full_data.iloc[tst_idx, :].to_csv(os.path.join(stgs.PRED_BATCH_PATH,'textfiles/full_test.csv'),index=False)
    


def scrolling_gen_batch(dir_path, tr_filename, tst_filename, win_size, win_pos, test_pct = 0.2):
    with open(os.path.join(dir_path,tr_filename), 'r') as f:
        tr_data = list(csv.reader(f))
    with open(os.path.join(dir_path,tst_filename), 'r') as f:
        tst_data = list(csv.reader(f))
    tr_opencsv=open(os.path.join(dir_path,"train.csv"),'w',newline='')
    tr_writer=csv.writer(tr_opencsv)
    tst_opencsv=open(os.path.join(dir_path,"test.csv"),'w',newline='')
    tst_writer=csv.writer(tst_opencsv)
    node_sizes = list(range(win_pos, win_pos + win_size))
    print(f'Networks depths considered: {node_sizes}.')
    tr_len, tst_len = 0, 0
    for row in tr_data:
        cnt = row[0].count('/') - 1
        if cnt in node_sizes:
            tr_writer.writerow(row)
            tr_len += 1
    tr_opencsv.close()
    for row in tst_data:
        cnt = row[0].count('/') - 1
        if cnt in node_sizes:
            tst_writer.writerow(row)
            tst_len += 1
    tst_opencsv.close()
    print(f"Training on {tr_len} samples, testing on {tst_len}.")
    return(tr_len,tst_len)
epoch=stgs.VAE_HPARAMS['max_steps']
time=1
p=0.4
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
            os.remove(os.path.join(path,file))

    stgs.LOG_PATH = Path('test_predictor/log')
 
    #stgs.WTS_PATH = Path('test_predictor/vae_wts/weights_2048.pt')

    stgs.PRED_BATCH_PATH = Path('test_predictor/pred_batches')
    stgs.VAE_HPARAMS['weights_path'] = Path("test_predictor/vae_wts/weights_256.pt")
    

    stgs.PRED_HPARAMS["weights_path"] = Path('test_predictor/integrated_pred_ckpts')

    # delete previous logs and windowbatch.csv
    # clear_files(stgs.LOG_PATH)
    #clear_files(stgs.PRED_BATCH_PATH,'textfiles/windowbatch.csv')

    
    g_mdl = NASGrammarModel(grammar, 'cuda')

    # 4. Train the predictor
    
    predictor = IntegratedPredictor(g_mdl,pretrained_vae=True,freeze_vae=False)
    
    logger = TensorBoardLogger(save_dir=stgs.LOG_PATH,
                               version='latent_sz_256_trained_unfrozen_lr1e-4_0')
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
   
    split_population(stgs.PRED_BATCH_PATH, 'textfiles/fitnessbatch.csv', test_pct=p)
    while window_pos + current_window_size - 1 <= max_depth:
        print('NEW BATCH SCROLL Window Size = ' + str(current_window_size) + ', Window Position = ' + str(window_pos))
    
        tr_len,tst_len=scrolling_gen_batch(stgs.PRED_BATCH_PATH, 'textfiles/full_train.csv', 'textfiles/full_test.csv',
                            current_window_size, window_pos, test_pct=p
                            )
       
        
       
        if current_window_size >= stgs.PRED_HPARAMS['warm_up_period']:  # let samples accumulate for a while
            trainer.fit(predictor)
            
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
            results.to_csv('test_predictor/exp_integrated.csv')
            # results.to_csv('../../test_predictor/exp_1.csv')
            print(f"Spearman correlation: {results.corr('spearman').iloc[0,1]:.4f}")

            #####################
            re=open(stgs.PRED_BATCH_PATH/'result/hvae.csv','a+',newline='')
            fieldnames = ['time','test_pro', 'depths','correlation',"train_num","test_num",'freeze','epoch']
            re_writer=csv.DictWriter(re, fieldnames=fieldnames)
            depths=list(range(window_pos,window_pos+current_window_size))
            window_size=current_window_size
            window_posistion=window_pos
            correlation=round(results.corr('spearman').iloc[0,1],4)
            re_writer.writerow({'time':time,'test_pro':p,
                                'depths':depths,
                                'correlation':correlation,
                                "train_num":tr_len,
                                "test_num":tst_len,
                                'freeze':'false',
                                'epoch':epoch})
            re.close()
            ##########################
            # prepare next iteration
            it += 1
            trainer = pl.Trainer(
                min_epochs = it * stgs.PRED_HPARAMS["max_epochs"],
                max_epochs = it * stgs.PRED_HPARAMS["max_epochs"],
                resume_from_checkpoint = os.path.join(stgs.PRED_HPARAMS["weights_path"],"predictor.ckpt"),
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



