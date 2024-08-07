import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

import numpy as np
import pandas as pd
import rasterio
import torch

from torch.utils.data import Dataset

from utils.utils import im_resize, im_crop


class BreizhSRDataset(Dataset):
    """Helper class for data preprocessing

    You can use this class to load the data, however it is recommended that you use
    instead the TorchBreizhSRDataset class for faster data loading.
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        max_s2_images: int = 1,
        return_type: str = "interpolated",
        sr_scale: int = 4,
    ):
        """Inputs:
        - path to root of x,y,.pkl
        - split: str either train or test
        - max_s2_images: int deciding how many sen2 are returned
        - sr_scale: int upsampling factor
        - return_type: either ["interpolated","cross_sensor"]
        """

        # Set args as object attributes
        self.dataset_root = dataset_root
        self.split = split
        self.max_s2_images = max_s2_images
        self.return_type = return_type
        self.sr_scale = sr_scale

        # check if all args are valid
        assert self.split in ["train", "test"]
        assert self.return_type in ["interpolated", "cross_sensor"]

        # Load DataFrame depending on the requested split
        if self.split == "train":
            self.dataset = pd.read_pickle(
                os.path.join(self.dataset_root, "dataset_train.pkl")
            )
        if self.split == "test":
            self.dataset = pd.read_pickle(
                os.path.join(self.dataset_root, "dataset_test.pkl")
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get row from DataFrame
        dataset_row = self.dataset.iloc[idx]

        # Extract temporal encodings of acquisitions dates for Sentinel-2
        # We use relative encodings, i.e. we subtract to the encoding the SPOT-6 acquisition date
        dates_encoding = [
            (date_sen2 - dataset_row.dates_spot6).days
            for date_sen2 in dataset_row.dates_sen2
        ]

        # Get SPOT-6 HR image path
        hr_path = os.path.join(self.dataset_root, dataset_row["spot6_image"])
        # Get Sen-2 LR image paths by prefixing filename with dataset root path
        lr_paths = {
            key: os.path.join(self.dataset_root, value)
            for key, value in dataset_row["sen2_acquisitions"].items()
        }

        # Read HR SPOT-6 image and convert to torch.Tensor
        with rasterio.open(hr_path) as hr_file:
            hr = torch.Tensor(hr_file.read())

        # Read LR Sentinel-2 image(s) into a list
        lr = []  # empty list to hold the images

        # TODO: document this properly, not sure I understand it yet
        if self.max_s2_images == 1:  # SISR
            lr_days_delta = np.sort(
                [int(x) for x in lr_paths.keys()]
            )  # get sorted list
        else:  # MISR
            lr_days_delta = [int(x) for x in lr_paths.keys()]

        for count, value in enumerate(lr_days_delta):
            # stop loop when wanted amount of images reached
            if len(lr) >= self.max_s2_images:
                break
            # read lr image from path and append to list
            with rasterio.open(lr_paths[value]) as im:
                lr_im = im.read()
                lr_im = torch.Tensor(lr_im)
                if lr_im.shape[1] == lr_im.shape[2]:
                    lr.append(lr_im)

        # If the list contains less than the number of Sentinel-2 images requested, we pad the list
        # by repeating its last entry. Padded images are denoted by alpha = 0
        alphas = [1 for i in range(len(lr))]
        while len(lr) < self.max_s2_images:
            lr.append(lr[-1])
            dates_encoding.append(dates_encoding[-1])
            alphas.append(0)

        # Perform standard preprocessing
        hr = (hr / 255).float()  # SPOT-6 preprocessing: divide by 255 to get into [0,1]
        lr = [
            (tensor / 10000).float() for tensor in lr
        ]  # Sentinel-2 preprocessing: divide by 10000 to get reflectances

        # Stack LR images alongside batch dimension
        lr = torch.stack(lr)
        lr = lr.view(-1, lr.size(2), lr.size(3))

        # Crop the center of the images to match the upsampling factor
        hr, lr = im_crop(hr, lr, self.sr_scale)
        # Apply a classical upsampling interpolation to the LR image
        img_lr_up = torch.tensor(im_resize(lr, self.sr_scale))

        if self.max_s2_images == 1:  # SISR
            dict_return = {
                "img_hr": hr,
                "img_lr": lr,
                "img_lr_up": img_lr_up,
                "item_name": str(idx),
            }
        else:  # MISR
            lr = lr.reshape([self.max_s2_images, 3, lr.shape[-1], lr.shape[-1]])
            img_lr_up = img_lr_up.reshape(
                [self.max_s2_images, 3, img_lr_up.shape[-1], img_lr_up.shape[-1]]
            )

            dict_return = {
                "img_hr": hr,
                "img_lr": lr,
                "img_lr_up": img_lr_up,
                "item_name": str(idx),
                "dates_encoding": dates_encoding[: self.max_s2_images],
                "alphas": alphas,
            }
        return dict_return
