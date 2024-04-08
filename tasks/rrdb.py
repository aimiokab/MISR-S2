import torch
import torch.nn.functional as F
from models.diffsr_modules import RRDBNet
from tasks.srdiff_celeb import CelebDataSet
from tasks.srdiff_df2k import Df2kDataSet
from tasks.srdiff_sat import SatDataSet
from utils.hparams import hparams
from tasks.trainer import Trainer

import numpy as np


class RRDBTask(Trainer):
    def build_model(self):
        hidden_size = hparams['hidden_size']
        self.model = RRDBNet(3, 3, hidden_size, hparams['num_block'], hidden_size // 2)
        return self.model

    def build_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=hparams['lr'])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, 75000, 0.5)

    def training_step(self, sample):
        img_hr = sample['img_hr']
        img_lr = sample['img_lr']
        p = self.model(img_lr)
        loss = F.l1_loss(p, img_hr, reduction='mean')
        return {'l': loss, 'lr': self.scheduler.get_last_lr()[0]}, loss

    def sample_and_test(self, sample):
        ret = {k: 0 for k in self.metric_keys}
        ret['n_samples'] = 0

        img_hr = sample['img_hr']
        img_lr = sample['img_lr']
        img_sr = self.model(img_lr)
        img_sr = img_sr.clamp(-1, 1)
        loss = F.l1_loss(img_sr, img_hr, reduction='mean')
        for b in range(img_sr.shape[0]):
            s = self.measure.measure(img_sr[b], img_hr[b], img_lr[b], hparams['sr_scale'])
            ret['psnr'].append(s['psnr'])
            ret['ssim'].append(s['ssim'])
            ret['lpips'].append(s['lpips'])
            ret['mae'].append(s['mae'])
            ret['mse'].append(s['mse'])
        return img_sr, img_sr, ret, loss


class RRDBTaskSat(Trainer):
    def build_model(self):
        hidden_size = hparams['hidden_size']
        self.model = RRDBNet(3, 3, hidden_size, hparams['num_block'], hidden_size // 2)
        self.global_step = 0
        return self.model

    def build_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=hparams['lr'])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, 75000, 0.5)

    def training_step(self, sample):
        img_hr = sample['img_hr']
        img_lr = sample['img_lr']
        p = self.model(img_lr)
        if hparams["loss_type"]=="l1":
            loss = F.l1_loss(p, img_hr, reduction='mean')
        elif hparams["loss_type"]=="l1_shift":
            loss = self.shift_l1_loss(img_hr, p)
        return {'l': loss, 'lr': self.scheduler.get_last_lr()[0]}, loss

    def sample_and_test(self, sample):
        ret = {k: [] for k in self.metric_keys}
        ret['n_samples'] = 0
        
        img_hr = sample['img_hr']
        img_lr = sample['img_lr']
        img_sr = self.model(img_lr)
        img_sr = img_sr.clamp(-1, 1)
        if hparams["loss_type"]=="l1":
            loss = F.l1_loss(img_sr, img_hr, reduction='mean')
        elif hparams["loss_type"]=="l1_shift":
            loss = self.shift_l1_loss(img_hr, img_sr)
        #print(img_hr.shape)
        #print("ok")
        for b in range(img_sr.shape[0]):
            s = self.measure.measure(img_sr[b], img_hr[b], img_lr[b], hparams['sr_scale'])

            imgA = np.round((img_sr[b].cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.int32)
            imgB = np.round((img_hr[b].cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.int32)
            imgA = imgA.transpose(1, 2, 0)
            imgB = imgB.transpose(1, 2, 0)
            print(imgA.shape)
            print(np.abs(imgA - imgB).mean())

            ret['psnr'].append(s['psnr'])
            ret['ssim'].append(s['ssim'])
            ret['lpips'].append(s['lpips'])
            ret['mae'].append(s['mae'])
            ret['mse'].append(s['mse'])
            ret['shift_mae'].append(s['shift_mae'])
            ret['n_samples'] += 1
        return img_sr, img_sr, ret, loss

    def shift_l1_loss(self, y_true, y_pred, border=3):
        """
        Modified l1 loss to take into account pixel shifts
        """
        max_pixels_shifts = 2*border
        size_image = y_true.shape[-1]
        patch_pred = y_pred[..., border:size_image - border,
                                 border:size_image - border]

        X = []
        for i in range(max_pixels_shifts+1):
            for j in range(max_pixels_shifts+1):
                patch_true = y_true[..., i:i+(size_image-max_pixels_shifts),
                                        j:j+(size_image-max_pixels_shifts)]
                l1_loss =(patch_true-patch_pred).abs().mean()
                X.append(l1_loss)
        X = torch.stack(X)
        min_l1 = X.min(dim=0).values
        return min_l1


class RRDBCelebTask(RRDBTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = CelebDataSet


class RRDBDf2kTask(RRDBTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = Df2kDataSet

class RRDBSatTask(RRDBTaskSat):
    def __init__(self):
        super().__init__()
        self.dataset_cls = None #SatDataSet