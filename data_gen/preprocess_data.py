import argparse
import click
import logging
import os
import pandas as pd
import torch

from tqdm import tqdm

from dataset import Dataset

def preprocess_and_save(path_to_dataset, split="train", max_s2_images=1):
    dataset_name = f"dataset_{split}.pkl"
    
    dict = {}
    # Single-image SR
    if max_s2_images == 1:
        sr_type = "SISR"
        dict_keys = ['path_hr', 'path_lr', 'path_lr_up']
        image_keys = ['img_hr', 'img_lr', 'img_lr_up']
    # Multi-image SR
    else:
        sr_type = "MISR"
        dict_keys = ['path_lr']
        image_keys = ['img_lr']

    output_folder = os.path.join(path_to_dataset, "preprocessed", sr_type, f"dataset_{split}")
    
    # Create subfolders for the input series and output ground truths
    for key in image_keys:
        subfolder_path = os.path.join(output_folder, key)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

    # Read dataframe and load dataset 
    dataset = Dataset(path_to_dataset, split=split, max_s2_images=max_s2_images)
    dataset_pkl = dataset.dataset

    
    alphas_dates = {'alphas': [], 'dates_encoding': []}

    for idx, data in enumerate(tqdm(dataset)):
        index = dataset_pkl.index[idx]
        
        if max_s2_images > 1:
            alphas_dates['alphas'].append(data["alphas"])
            alphas_dates['dates_encoding'].append(data["dates_encoding"])

        # Unpack and save torch Tensors on disk
        for key in image_keys:
            tensor_path = os.path.join(output_folder, key, f"index.pt")
            torch.save(data[key], tensor_path)
    
    for i, key in enumerate(dict_keys):
        dict[key] = [os.path.join('',*(type_path[1:]+[image_keys[i],str(x)+".pt"])) for x in indexes]

    dict["index"] = dataset_pkl.index

    # Create new DataFrame for preprocessed dataset
    df = pd.DataFrame(dict).set_index("index")
    df = dataset_pkl.join(df)

    if max_s2_images > 1:  # MISR, add columns for date encoding
        df = df.assign(alphas = alphas_dates['alphas'])
        df = df.assign(dates_encoding = alphas_dates['dates_encoding'])
        df = df.rename(columns={'path_lr': 'path_lr_misr'})
    else: # SISR
        df = df.rename(columns={'path_lr': 'path_lr_sisr'})
    return df

@click.command()
@click.argument('path_to_dataset', type=click.Path(exists=True))
#@click.option('--split', type=click.Choice(['train', 'test'], case_sensitive=False), help="Split to preprocess (either train or test)")
@click.option('--max_s2_images', type=int, help="Maximum number of Sentinel-2 images to include in the input series (use 1 for single-image super-resolution)")
def preprocess_dataset(path_to_dataset, max_s2_images=1):
    # Preprocessing SISR train set
    df_train_sisr = preprocess_and_save(path_to_dataset, "train", 1)
    # Preprocessing SISR test set
    df_test_sisr = preprocess_and_save(path, "test", 1)
    # Preprocessing MISR train set
    df_train_misr = preprocess_and_save(path, "train", max_s2_images)
    # Preprocessing MISR test set
    df_test_misr = preprocess_and_save(path, "test", max_s2_images)


    # Join DataFrames
    df_train = df_train_sisr.join(df_train_misr[["path_lr_misr","alphas","dates_encoding"]])
    df_test = df_test_sisr.join(df_test_misr[["path_lr_misr","alphas","dates_encoding"]])

    # Save DataFrames on disk in `preprocessed` subfolder
    df_train.to_pickle(os.path.join(path, 'preprocessed', 'dataset_train.pkl'))
    df_test.to_pickle(os.path.join(path, 'preprocessed', 'dataset_test.pkl'))

if __name__ == '__main__':
    preprocess_dataset()
