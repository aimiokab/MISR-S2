import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from utils.dataloader_utils import stretch_standardize, minmax

import pandas as pd
import numpy as np
import os

from utils.hparams import hparams
import random


# Define torch dataset Class
class Dataset(Dataset):
    def __init__(self, dataset_root, phase="train", sen2_amount=1):
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
        
        # load dataset file

        if self.phase in ["train","val"]:
            self.dataset = pd.read_pickle(os.path.join('', *[self.dataset_root, "preprocessed", "dataset_train.pkl"]))
        if self.phase == "test" :
            dataset = pd.read_pickle(os.path.join('', *[self.dataset_root, "preprocessed", "dataset_test.pkl"]))

        # filter for phase
        #self.dataset = dataset[dataset["type_split"] == self.phase]

        if self.phase == "test":
            self.dataset = self.dataset.iloc[20:50]


        #self.dataset = self.dataset.iloc[:128]
        #self.dataset = self.dataset.loc[self.dataset.index.isin(random.sample(list(self.dataset.index),1000))]

        self.items = self.dataset.index.astype(str)

        self.to_tensor_norm = stretch_standardize

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

        #lr = lr[:,:40,:40]
        #hr = hr[:,:160,:160]  #p = x.unfold(1, size, stride).unfold(2, size, stride)
        #img_lr_up = img_lr_up[:,:160,:160]

        if self.sen2_amount==1:
            #hr, lr, img_lr_up = [self.to_tensor_norm(x) for x in [hr, lr, img_lr_up]]
            #hr = self.to_tensor_norm(hr, "spot6")
            #lr = self.to_tensor_norm(lr, "sen2")
            #img_lr_up = self.to_tensor_norm(img_lr_up, "sen2")

            dict_return = {
                        'img_hr': hr, 'img_lr': lr,
                        'img_lr_up': img_lr_up,'item_name': str(idx),
                    }
        else:
            #hr = self.to_tensor_norm(hr, "spot6", "misr")
            #img_lr_up = self.to_tensor_norm(img_lr_up, "sen2", "misr")

            #for i in range(self.sen2_amount):
            #    lr[i] = self.to_tensor_norm(lr[i], "sen2")
            
            dict_return = {
                        'img_hr': hr, 'img_lr': lr[:self.sen2_amount,...],
                        'img_lr_up': img_lr_up[:self.sen2_amount,...],'item_name': str(idx), 'dates_encoding': dates_encoding[:self.sen2_amount], 'alphas': alphas[:self.sen2_amount]
                    }
        return dict_return