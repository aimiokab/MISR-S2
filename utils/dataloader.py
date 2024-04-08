import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from utils.dataloader_utils import stretch_standardize, minmax, histogram

import pandas as pd
import numpy as np
import os

from utils.hparams import hparams
import random

from utils import dataloader_utils
from sklearn.utils import shuffle
from utils.utils import im_resize

# Define torch dataset Class
class Dataset(Dataset):
    def __init__(self, dataset_root, phase="train", sen2_amount=1, spectral_matching="histogram"):
        """  Inputs:
                - path to root of x,y,.pkl
                - phase: str either train or test
                - spectral_matching: str either ["histogram","moment","normalize"]
                - spatial_matching: str either ["grid","shiftnet","none"]
                - sen2_amount: int deciding how many sen2 are returned
                - return_type: either ["interpolated","cross_sensor"]
        """
        
        # set args as class props
        self.dataset_root = dataset_root
        self.phase = phase
        self.sen2_amount = sen2_amount
        self.spectral_matching = spectral_matching
        
        # load dataset file

        if self.phase in ["train","val"]:
            self.dataset = pd.read_pickle(os.path.join('', *[self.dataset_root, "preprocessed", "dataset_train.pkl"]))
        if self.phase == "test" :
            self.dataset = pd.read_pickle(os.path.join('', *[self.dataset_root, "preprocessed", "dataset_test.pkl"]))

        # filter for phase
        self.dataset = self.dataset[self.dataset["type_split"] == self.phase]

        
        if self.phase =="test" and hparams["subset_test"]:
            random.seed(45)
            self.dataset = self.dataset.loc[self.dataset.index.isin(random.sample(list(self.dataset.index),100))]
            
        self.dataset['indexes'] = self.dataset.index

        # load spectral matching from file
        self.spectral_matching = getattr(dataloader_utils,self.spectral_matching)

        self.items = self.dataset.index.astype(str)

        self.to_tensor_norm = stretch_standardize

        if not sen2_amount==1:
            self.dataset["nb_images"] = self.dataset.dates_encoding.apply(lambda x: np.argsort(np.abs(x))[:sen2_amount])


    def __len__(self):
        return(len(self.dataset))
    
    def __getitem__(self,idx):
        
        # get row from dataset file
        dataset_row = self.dataset.iloc[idx]

        # get HR image path
        hr_path = os.path.join(self.dataset_root,dataset_row["path_hr"])
        # get LR image paths
        lr_up_path = os.path.join(self.dataset_root,dataset_row["path_lr_up"])

        if self.sen2_amount>1:
            lr_path  =  os.path.join(self.dataset_root,dataset_row["path_lr_misr"])
            dates_encoding = dataset_row['dates_encoding']
            alphas = dataset_row['alphas']
        else :
            lr_path  =  os.path.join(self.dataset_root,dataset_row["path_lr_sisr"])

        hr = torch.load(hr_path)
        lr = torch.load(lr_path)
        img_lr_up = torch.load(lr_up_path)


        if self.sen2_amount==1:
            hr = self.to_tensor_norm(hr, "spot6")
            lr = self.to_tensor_norm(lr, "sen2")
            img_lr_up = self.to_tensor_norm(img_lr_up, "sen2")

            hr = self.spectral_matching(lr,hr)

            dict_return = {
                        'img_hr': hr, 'img_lr': lr,
                        'img_lr_up': img_lr_up,'item_name': str(idx),
                    }
        else:
            lr = lr[dataset_row["nb_images"],:]
            dates_encoding = [dates_encoding[x] for x in dataset_row["nb_images"]]
            hr = self.to_tensor_norm(hr, "spot6")
            img_lr_up = self.to_tensor_norm(img_lr_up, "sen2")

            lr = self.to_tensor_norm(lr, "sen2","misr")
            ind = np.argmin(np.abs(dates_encoding))
            hr = self.spectral_matching(lr[ind].squeeze(),hr)
            
            dict_return = {
                        'img_hr': hr, 'img_lr': lr[:self.sen2_amount,...],
                        'img_lr_up': img_lr_up[:self.sen2_amount,...],'item_name': str(idx), 'dates_encoding': torch.Tensor(dates_encoding), 'alphas': torch.Tensor(alphas[:self.sen2_amount]), 'indices': ind
                    }
        if self.phase == "test":
            dict_return['indexes'] = dataset_row['indexes']
        return dict_return