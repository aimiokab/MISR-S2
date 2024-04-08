import torch
import numpy as np
import sys
sys.path.insert(1, '/share/projects/sesure/aimi/SRDiff_test/')
from utils.dataloader import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import Measure
import pandas as pd
import argparse
import os



def build_test_dataloader(subset=False):
    dataset_test = Dataset(path,phase="test",sen2_amount=1)
    dataloader = DataLoader(dataset_test,batch_size=1,
                        shuffle=False, num_workers=0,pin_memory=True,drop_last=True)    
    return dataloader


def test_bicubic():
    test_dataloader = build_test_dataloader()
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    metrics = ['psnr', 'ssim', 'lpips', 'mae', 'mse', 'shift_mae']
    measure = Measure()
    test_results = {k: [] for k in metrics}
    test_results['key'] = []
    for batch_idx, batch in pbar:
        item_names = batch['item_name']
        img_hr = batch['img_hr']
        img_lr = batch['img_lr']
        img_lr_up = batch['img_lr_up']
        ret = {k: [] for k in ['psnr', 'ssim', 'lpips', 'lr_psnr', 'mae', 'mse', 'shift_mae']}
        ret['n_samples'] = 0
        for b in range(img_lr_up.shape[0]):
            s = measure.measure(img_lr_up[b], img_hr[b], img_lr[b], 4)

            ret['psnr'].append(s['psnr'])
            ret['ssim'].append(s['ssim'])
            ret['lpips'].append(s['lpips'])
            ret['mae'].append(s['mae'])
            ret['mse'].append(s['mse'])
            ret['shift_mae'].append(s['shift_mae'])
            ret['n_samples'] += 1
        test_results['key'].append(batch['indexes'].cpu().numpy())
        for k in metrics:
            test_results[k].append(np.mean(ret[k]))
    res = pd.DataFrame(test_results)
    res.to_csv(path+"test_results_bicubic.csv", sep=";")

if __name__ == '__main__':
    path = "/share/projects/sesure/aimi/data/" 
    test_bicubic()