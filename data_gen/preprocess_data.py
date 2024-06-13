import os
import pandas as pd
from tqdm import tqdm
import argparse
import torch
from dataset import Dataset

def preprocess_and_save(path, phase="train", sen2_amount=1, dataset_folder="BreizhSR"):
    dataset_name = "dataset_"+phase+".pkl"
    
    if sen2_amount==1:
        dict = {}
        dict_keys = ['path_lr', 'path_hr', 'path_lr_up']
        type = "SISR"
        imgs = ['img_hr', 'img_lr', 'img_lr_up']
    else:
        dict = {}
        dict_keys = ['path_lr']
        type = "MISR"
        imgs = ['img_lr']
    type_path = [path, "preprocessed", type, "dataset_"+phase]
    path_save = os.path.join('', *type_path)
    
    for key in imgs:
        sub_path = os.path.join(path_save, key)
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
    dataset_pkl = pd.read_pickle(os.path.join(path, dataset_folder,dataset_name))
    dataset= Dataset(os.path.join(path, dataset_folder),phase=phase,sen2_amount=sen2_amount)

    
    pbar = tqdm(dataset)
    alphas_dates = {'alphas': [], 'dates_encoding': []}

    indexes = dataset_pkl.index

    for e,data in enumerate(pbar):

        index = indexes[e]
        
        if sen2_amount > 1:
            alphas_dates['alphas'].append(data["alphas"])
            alphas_dates['dates_encoding'].append(data["dates_encoding"])
        for key in imgs:
            p = [path_save, key, str(index)+".pt"]
            new_p = os.path.join('', *p)

            torch.save(data[key], new_p)
    
    for i,key in enumerate(dict_keys):
        dict[key] = [os.path.join('',*(type_path[1:]+[imgs[i],str(x)+".pt"])) for x in indexes]
    dict["index"] = indexes
    df = pd.DataFrame(dict).set_index("index")

    df = dataset_pkl.join(df)

    if sen2_amount>1:
        df = df.assign(alphas = alphas_dates['alphas'])
        df = df.assign(dates_encoding = alphas_dates['dates_encoding'])
        df = df.rename(columns={'path_lr': 'path_lr_misr'})
    else:
        df = df.rename(columns={'path_lr': 'path_lr_sisr'})
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocessing')
    parser.add_argument("--sen2_amount", help='number of images')
    args = parser.parse_args()

    path = "/share/projects/sesure/aimi/data"

    df_train_sisr = preprocess_and_save(path, "train", 1)
    df_test_sisr = preprocess_and_save(path, "test", 1)
    df_train_misr = preprocess_and_save(path, "train", int(args.sen2_amount))
    df_test_misr = preprocess_and_save(path, "test", int(args.sen2_amount))
    df_train = df_train_sisr.join(df_train_misr[["path_lr_misr","alphas","dates_encoding"]])
    df_test = df_train_sisr.join(df_test_misr[["path_lr_misr","alphas","dates_encoding"]])

    df_train.to_pickle(os.path.join(path, 'preprocessed', 'dataset_train.pkl'))
    df_test.to_pickle(os.path.join(path, 'preprocessed', 'dataset_test.pkl'))