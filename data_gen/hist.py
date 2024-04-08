import os
import sys
#sys.path.insert(1, '/home/aimi/Documents/code/tests/SRDiff_test/')
sys.path.insert(1, '/share/projects/sesure/aimi/SRDiff_test/')
import pandas as pd
from tqdm import tqdm
import argparse
import torch
from dataset import Dataset
import sys
from utils.dataloader_hist import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random


def cumulative_histogram(path, phase="train", sen2_amount=1):

    s = 74*74
    subset = np.random.choice(s, s//5, replace=False)
    
    dataset = Dataset(path,phase="train", sen2_amount=sen2_amount)
    pbar = tqdm(dataset)
    dict = {x: [] for x in range(3)}
    for e,data in enumerate(pbar):
        sen2 = data["img_lr"]
        for i in range(3):
            dict[i] += list(sen2[i].flatten().numpy()[subset])
    hist = pd.DataFrame(dict)
    hist = hist.drop(hist.loc[hist[0]==-1].index)
    hist.to_pickle(os.path.join(path, "cumhist_subset.pkl"))
    #hist = pd.read_pickle(os.path.join(path,"cumhist.pkl"))
    hist = hist.reset_index(drop=True)

    #subset = np.random.choice(203549371, 203549371//10, replace=False)

    c=["r","g","b"]
    labels=['R','G','B']

    quantiles = {x: hist.quantile(x) for x in [0.95, 0.96, 0.97, 0.98, 0.99]}

    #hist = hist.loc[hist.index.isin(subset)]

    for i in range(3):
        plt.figure()
        sns.kdeplot(hist[i], c=c[i])
        plt.title(labels[i])
        
        #l = [hist.quantile(x)[i] for x in [0.95, 0.96, 0.97, 0.98, 0.99]]
        #for qt in l:
        #    plt.axvline(qt, label=str(np.round(qt,3)))
        for x in quantiles.keys():
            plt.axvline(quantiles[x][i], label=str(np.round(quantiles[x][i],3)))

        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(path,str(i)+"_distrib.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'preprocessing')

    parser.add_argument("--phase", help='phase: train or test')
    parser.add_argument("--sen2_amount", help='number of images')
    args = parser.parse_args()

    path = "/share/projects/sesure/aimi/data/" #"/home/aimi/Documents/code/tests/data/"

    cumulative_histogram(path, args.phase, int(args.sen2_amount))