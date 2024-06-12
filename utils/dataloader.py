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
from utils.utils import im_resize

# Define torch dataset Class
class BreizshSRDataset(Dataset):
    def __init__(self, dataset_root : str, phase : str = "train", sen2_amount : int = 1, spectral_matching : str = "histogram"):
        """  Inputs:
                - path to root of x,y,.pkl
                - phase: str either train or test
                - spectral_matching: str either ["histogram","moment","normalize"]
                - spatial_matching: str either ["grid","shiftnet","none"]
                - sen2_amount: int deciding how many sen2 are returned
                - return_type: either ["interpolated","cross_sensor"]
        """
        
        # Set args as object attributes
        self.dataset_root = dataset_root
        self.phase = phase
        self.sen2_amount = sen2_amount
        self.spectral_matching = spectral_matching

        # Check that required files are here
        preprocessed_path = os.path.join(self.dataset_root, "preprocessed")
        if not os.path.isdir(preprocessed_path):
            raise Exception("There should be a ``preprocessed`` folder at {self.dataset_root}. Maybe you didn't run the preprocessing script?")
        
        # Read dataset DataFrame from disk
        if self.phase in ["train", "val"]:
            self.dataset = pd.read_pickle(os.path.join(preprocessed_path, "dataset_train.pkl"))
        if self.phase == "test" :
            self.dataset = pd.read_pickle(os.path.join(preprocessed_path, "dataset_test.pkl"))

        # Keep only the rows from the request split
        self.dataset = self.dataset[self.dataset["type_split"] == self.phase]

        if self.phase == "test" and hparams["subset_test"]:
            random.seed(45) # FIXME
            self.dataset = self.dataset.loc[self.dataset.index.isin(random.sample(list(self.dataset.index),100))] # FIXME ??
        
        self.dataset['indexes'] = self.dataset.index

        # Load the correct spectral matching function from the utils module
        self.spectral_matching = getattr(dataloader_utils, self.spectral_matching)

        self.items = self.dataset.index.astype(str)

        self.to_tensor_norm = stretch_standardize

        # In case of multi-image super-resolution, create a column representing the number of images for each series
        # `sen2_amount` defines the maximum number of S2 images that are used
        if sen2_amount > 1:
            self.dataset["nb_images"] = self.dataset.dates_encoding.apply(lambda x: np.argsort(np.abs(x))[:sen2_amount])


    def __len__(self) -> int:
        return(len(self.dataset))
    
    def __getitem__(self, idx : int):
        
        # Get row from DataFrame
        dataset_row = self.dataset.iloc[idx]

        # Get path to HR image
        hr_path = os.path.join(self.dataset_root, dataset_row["path_hr"])
        # Get path to (upsampled) LR image
        lr_up_path = os.path.join(self.dataset_root, dataset_row["path_lr_up"])

        if self.sen2_amount > 1: # MISR
            lr_path = os.path.join(self.dataset_root, dataset_row["path_lr_misr"])
            dates_encoding = dataset_row['dates_encoding']
            alphas = dataset_row['alphas']
        else: # SISR
            lr_path = os.path.join(self.dataset_root, dataset_row["path_lr_sisr"])

        # Read tensors on disk
        hr = torch.load(hr_path) # HR ground truth
        lr = torch.load(lr_path) # LR image
        img_lr_up = torch.load(lr_up_path) # Upsampled LR image

        if self.sen2_amount == 1:
            # Normalize tensors
            hr = self.to_tensor_norm(hr, "spot6")
            lr = self.to_tensor_norm(lr, "sen2")
            img_lr_up = self.to_tensor_norm(img_lr_up, "sen2")

            # Match the ground truth radiometry to the reference low resolution input
            hr = self.spectral_matching(lr, hr)

            dict_return = {
                        'img_hr': hr,
                        'img_lr': lr,
                        'img_lr_up': img_lr_up,
                        'item_name': str(idx),
                    }
        else:
            lr = lr[dataset_row["nb_images"], :]
            # Extract positional encodings for dates
            dates_encoding = [dates_encoding[x] for x in dataset_row["nb_images"]]
            # Normalize tensors
            hr = self.to_tensor_norm(hr, "spot6")
            img_lr_up = self.to_tensor_norm(img_lr_up, "sen2")
            lr = self.to_tensor_norm(lr, "sen2","misr")

            ind = np.argmin(np.abs(dates_encoding))

            # Match the ground truth radiometry to the reference low resolution input
            # (reference LR inpt is the closest image to the HR)
            hr = self.spectral_matching(lr[ind].squeeze(), hr)
            
            dict_return = {
                        'img_hr': hr, 'img_lr': lr[:self.sen2_amount,...],
                        'img_lr_up': img_lr_up[:self.sen2_amount,...],
                        'item_name': str(idx),
                        'dates_encoding': torch.Tensor(dates_encoding),
                        'alphas': torch.Tensor(alphas[:self.sen2_amount]), 
                        'indices': ind,
                    }

        if self.phase == "test":
            dict_return['indexes'] = dataset_row['indexes']
        return dict_return
