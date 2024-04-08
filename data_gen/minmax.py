import torch
import pandas as pd
import os
import argparse


def read_minmax(path, phase):
    dataset = pd.read_pickle(os.path.join('',*[path, "preprocessed","dataset_"+phase+".pkl"]))

    res = {**{"lr_min_"+str(x):[] for x in range(3)}, **{"lr_max_"+str(x):[] for x in range(3)}, **{"hr_min_"+str(x):[] for x in range(3)}, **{"hr_max_"+str(x):[] for x in range(3)}}
    res["index"] = []
    for index, row in dataset.iterrows():
        hr_path = os.path.join(path, row["path_hr"])
        lr_path  =  os.path.join(path, row["path_lr_sisr"])

        hr = torch.load(hr_path)
        lr = torch.load(lr_path)

        res['index'].append(index)
        for i in range(3):
            res["hr_min_"+str(i)].append(hr[i].min().numpy())
            res["hr_max_"+str(i)].append(hr[i].max().numpy())
            res["lr_min_"+str(i)].append(lr[i].min().numpy())
            res["lr_max_"+str(i)].append(lr[i].max().numpy())

    df = pd.DataFrame(res)
    df.to_pickle(os.path.join('', *[path,"distribution_"+phase+".pkl"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'preprocessing')

    parser.add_argument("--phase", help='phase: train or test')
    args = parser.parse_args()

    path = "/share/projects/sesure/aimi/data/"

    read_minmax(path, args.phase)