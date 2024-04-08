import torch
import torch.nn.functional as F
from models.misr_module import RRDBLtaeNet
from utils.hparams import hparams
from tasks.trainer import Trainer
import json


class RRDBLtae(Trainer):
    def build_model(self):
        hidden_size = hparams['hidden_size']
        with open("./tasks/config_rrdb_misr.json", "r") as read_file:
            self.config = json.load(read_file)
        self.model = RRDBLtaeNet(3, 3, hidden_size, hparams['num_block'], hidden_size // 2)
        self.global_step = 0
        return self.model

    def build_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=hparams['lr'])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, 100000, 0.5)


    def training_step(self, sample):
        img_hr = sample['img_hr']
        img_lr = sample['img_lr']
        dates = sample['dates_encoding']
        #print(img_lr.shape)
        p = self.model(img_lr, dates, self.config)
        print(p.shape)
        print(img_hr.shape)
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



