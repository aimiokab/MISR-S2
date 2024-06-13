import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import rasterio

from utils import dataloader_utils
from utils.utils import im_resize, im_crop

class Dataset(Dataset):
    def __init__(self,dataset_root,phase="train",spectral_matching="histogram",
                     spatial_matching="grid",sen2_amount=1,return_type="interpolated", subset=True, sr_scale=4):
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
        self.spectral_matching = spectral_matching
        self.spatial_matching = spatial_matching
        self.phase = phase
        self.sen2_amount = sen2_amount
        self.return_type = return_type
        self.sr_scale = sr_scale

        # check if all args are valid
        assert self.spectral_matching in ["histogram","moment"]
        assert self.spatial_matching in ["grid","shiftnet"]
        assert self.phase in ["train","test","val"]
        assert self.return_type in ["interpolated","cross_sensor"]
        
        # load dataset file
        if self.phase in ["train","val"]:
            self.dataset = pd.read_pickle(os.path.join(self.dataset_root,"dataset_train.pkl"))
        if self.phase == "test" :
            self.dataset = pd.read_pickle(os.path.join(self.dataset_root,"dataset_test.pkl"))
        
        # filter for phase
        #self.dataset = dataset[dataset["type_split"] == self.phase]

        # load spectral matching from file
        self.spectral_matching = getattr(dataloader_utils,self.spectral_matching)
        # load spatial_matching from file
        self.spatial_matching = getattr(dataloader_utils,self.spatial_matching)

        self.size_dataset = len(self.dataset)
        self.items = self.dataset.index.astype(str)

        self.to_tensor_norm = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        
    def __len__(self):
        return(len(self.dataset))
    
    def __getitem__(self,idx):
        
        # get row from dataset file
        dataset_row = self.dataset.iloc[idx]

        dates_encoding = [(x - dataset_row.dates_spot6).days for x in dataset_row.dates_sen2]
        
        # get HR image path
        hr_path = os.path.join(self.dataset_root,dataset_row["spot6_image"])
        # get LR image paths by adding dataset root path infront of dict values
        lr_paths  = {key: os.path.join(self.dataset_root,value) for key, value in dataset_row["sen2_acquisitions"].items()}
        
        # read HR image
        with rasterio.open(hr_path) as hr_file:
            hr = torch.Tensor(hr_file.read())
        
        # read lr image(s) into list
        lr = [] # empty list to hold images
        if self.sen2_amount==1:
            lr_days_delta = np.sort([int(x) for x in lr_paths.keys()]) # get sorted list
        else:
            lr_days_delta = [int(x) for x in lr_paths.keys()]
        for count,value in enumerate(lr_days_delta):
            # stop loop when wanted amount of images reached
            if len(lr)>=self.sen2_amount: 
                break
            #read lr image from path and append to list
            with rasterio.open(lr_paths[value]) as im:
                lr_im = im.read()
                lr_im = torch.Tensor(lr_im)
                if lr_im.shape[1]==lr_im.shape[2]:
                    lr.append(lr_im)
        
        # if there are not as many sen2 images as requested, repeat last entry in the list
        alphas = [1 for i in range(len(lr))]
        while len(lr) < self.sen2_amount: 
            lr.append(lr[-1])
            dates_encoding.append(dates_encoding[-1])
            alphas.append(0)
            
        # perform standard preprocessing
        hr = (hr/255).float()
        lr = [(tensor/10000).float() for tensor in lr]
        
        # perform spectral matching to closest sen2 image in time
        hr = self.spectral_matching(lr[0],hr)
        
        # stack lr to batch dimensions
        lr = torch.stack(lr)
        lr = lr.view(-1, lr.size(2), lr.size(3))
        hr, lr = im_crop(hr, lr, self.sr_scale)
        
        img_lr_up = torch.tensor(im_resize(lr, self.sr_scale))

        if self.sen2_amount==1:
            dict_return = {
                        'img_hr': hr, 'img_lr': lr,
                        'img_lr_up': img_lr_up,'item_name': str(idx),
                    }
        else:
            lr = lr.reshape([self.sen2_amount,3,lr.shape[-1],lr.shape[-1]])
            img_lr_up = img_lr_up.reshape([self.sen2_amount,3,img_lr_up.shape[-1],img_lr_up.shape[-1]])

            dict_return = {
                        'img_hr': hr, 'img_lr': lr,
                        'img_lr_up': img_lr_up,'item_name': str(idx), 'dates_encoding': dates_encoding[:self.sen2_amount], 'alphas': alphas
                    }
        return dict_return