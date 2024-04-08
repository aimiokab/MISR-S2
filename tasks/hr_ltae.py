import torch
import torch.nn.functional as F
from models.misr_module import HighResLtaeNet
from utils.hparams import hparams
from tasks.trainer import Trainer
import json


class MISR(Trainer):
    def build_model(self):
        with open("./tasks/config_hrnet.json", "r") as read_file:
            self.config = json.load(read_file)
        #{"in_channels": 3*2, "num_layers":10, "kernel_size":3, "channel_size":64}
        self.model = HighResLtaeNet(self.config)
        self.global_step = 0
        return self.model

    def build_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), hparams['lr'])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, 50000, 0.7)

    def training_step(self, sample):
        img_hr = sample['img_hr']
        img_lr = sample['img_lr']
        dates = sample['dates_encoding']
        p = self.model(img_lr, dates, self.config)
        loss = F.l1_loss(p, img_hr, reduction='mean')
        
        return {'l': loss, 'lr': self.scheduler.get_last_lr()[0]}, loss

    def sample_and_test(self, sample):
        ret = {k: [] for k in self.metric_keys}
        ret['n_samples'] = 0
        img_hr = sample['img_hr']
        img_lr = sample['img_lr']
        dates = sample['dates_encoding']
        img_sr = self.model(img_lr, dates, self.config)
        loss = F.l1_loss(img_sr, img_hr, reduction='mean')
        for b in range(img_sr.shape[0]):
            s = self.measure.measure(img_sr[b], img_hr[b], img_lr[b], hparams['sr_scale'])
            ret['psnr'].append(s['psnr'])
            ret['ssim'].append(s['ssim'])
            ret['lpips'].append(s['lpips'])
            ret['mae'].append(s['mae'])
            ret['mse'].append(s['mse'])
            ret['shift_mae'].append(s['shift_mae'])
            ret['n_samples'] += 1
        
        return img_sr, img_sr, ret, loss