import subprocess
import torch.distributed as dist
import glob
import os
import re
import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from .matlab_resize import imresize
import cv2 as cv


def reduce_tensors(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            dist.all_reduce(v)
            v = v / dist.get_world_size()
        if type(v) is dict:
            v = reduce_tensors(v)
        new_metrics[k] = v
    return new_metrics


def tensors_to_scalars(tensors):
    if isinstance(tensors, torch.Tensor):
        tensors = tensors.item()
        return tensors
    elif isinstance(tensors, dict):
        new_tensors = {}
        for k, v in tensors.items():
            v = tensors_to_scalars(v)
            new_tensors[k] = v
        return new_tensors
    elif isinstance(tensors, list):
        return [tensors_to_scalars(v) for v in tensors]
    else:
        return tensors


def tensors_to_np(tensors):
    if isinstance(tensors, dict):
        new_np = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = tensors_to_np(v)
            new_np[k] = v
    elif isinstance(tensors, list):
        new_np = []
        for v in tensors:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = tensors_to_np(v)
            new_np.append(v)
    elif isinstance(tensors, torch.Tensor):
        v = tensors
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if type(v) is dict:
            v = tensors_to_np(v)
        new_np = v
    else:
        raise Exception(f'tensors_to_np does not support type {type(tensors)}.')
    return new_np


def move_to_cpu(tensors):
    ret = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu()
        if type(v) is dict:
            v = move_to_cpu(v)
        ret[k] = v
    return ret


def move_to_cuda(batch, gpu_id=0):
    # base case: object can be directly moved using `cuda` or `to`
    if callable(getattr(batch, 'cuda', None)):
        return batch.cuda(gpu_id, non_blocking=True)
    elif callable(getattr(batch, 'to', None)):
        return batch.to(torch.device('cuda', gpu_id), non_blocking=True)
    elif isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return batch
    elif isinstance(batch, tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return tuple(batch)
    elif isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = move_to_cuda(v, gpu_id)
        return batch
    return batch


def get_last_checkpoint(work_dir, steps=None):
    checkpoint = None
    last_ckpt_path = None
    ckpt_paths = get_all_ckpts(work_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch.load(last_ckpt_path, map_location='cpu')
    return checkpoint, last_ckpt_path


def get_all_ckpts(work_dir, steps=None):
    if steps is None:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_*.ckpt'
    else:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_{steps}.ckpt'
    return sorted(glob.glob(ckpt_path_pattern),
                  key=lambda x: -int(re.findall('.*steps\_(\d+)\.ckpt', x)[0]))


def load_checkpoint(model, optimizer, work_dir):
    checkpoint, _ = get_last_checkpoint(work_dir)
    

    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict']['model'])
        model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer_states'][0])
        training_step = checkpoint['global_step']
        del checkpoint
        torch.cuda.empty_cache()
    else:
        training_step = 0
        model.cuda()
    return training_step


def save_checkpoint(model, optimizer, work_dir, global_step, num_ckpt_keep):
    ckpt_path = f'{work_dir}/model_ckpt_steps_{global_step}.ckpt'
    print(f'Step@{global_step}: saving model to {ckpt_path}')
    checkpoint = {'global_step': global_step}
    optimizer_states = []
    optimizer_states.append(optimizer.state_dict())
    checkpoint['optimizer_states'] = optimizer_states
    checkpoint['state_dict'] = {'model': model.state_dict()}
    torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False)
    for old_ckpt in get_all_ckpts(work_dir)[num_ckpt_keep:]:
        remove_file(old_ckpt)
        print(f'Delete ckpt: {os.path.basename(old_ckpt)}')


def remove_file(*fns):
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)


def plot_img(img):
    img = img.data.cpu().numpy()
    return np.clip(img, 0, 1)


def load_ckpt(cur_model, ckpt_base_dir, model_name='model', force=True, strict=True):
    if os.path.isfile(ckpt_base_dir):
        base_dir = os.path.dirname(ckpt_base_dir)
        ckpt_path = ckpt_base_dir
        checkpoint = torch.load(ckpt_base_dir, map_location='cpu')
    else:
        base_dir = ckpt_base_dir
        checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir)
    if checkpoint is not None:
        state_dict = checkpoint["state_dict"]
        if len([k for k in state_dict.keys() if '.' in k]) > 0:
            state_dict = {k[len(model_name) + 1:]: v for k, v in state_dict.items()
                          if k.startswith(f'{model_name}.')}
        else:
            state_dict = state_dict[model_name]
        if not strict:
            cur_model_state_dict = cur_model.state_dict()
            unmatched_keys = []
            for key, param in state_dict.items():
                if key in cur_model_state_dict:
                    new_param = cur_model_state_dict[key]
                    if new_param.shape != param.shape:
                        unmatched_keys.append(key)
                        print("| Unmatched keys: ", key, new_param.shape, param.shape)
            for key in unmatched_keys:
                del state_dict[key]
        cur_model.load_state_dict(state_dict, strict=strict)
        print(f"| load '{model_name}' from '{ckpt_path}'.")
    else:
        e_msg = f"| ckpt not found in {base_dir}."
        if force:
            assert False, e_msg
        else:
            print(e_msg)


class Measure:
    def __init__(self, net='alex'):
        self.model = lpips.LPIPS(net=net)

    def measure(self, imgA, imgB, img_lr, sr_scale):
        """

        Args:
            imgA: [C, H, W] uint8 or torch.FloatTensor [-1,1]
            imgB: [C, H, W] uint8 or torch.FloatTensor [-1,1]
            img_lr: [C, H, W] uint8  or torch.FloatTensor [-1,1]
            sr_scale:

        Returns: dict of metrics

        """
        
        if isinstance(imgA, torch.Tensor):
            imgA = np.round((imgA.cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.int32)
            imgB = np.round((imgB.cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.int32)
            img_lr = np.round((img_lr.cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.int32)
        imgA = imgA.transpose(1, 2, 0)
        imgA_lr = imresize(imgA, 1 / sr_scale)
        imgB = imgB.transpose(1, 2, 0)
        print(self.psnr(imgA, imgB))
        #img_lr = img_lr.transpose(1, 2, 0)
        psnr = self.psnr(imgA, imgB)
        ssim = self.ssim(imgA, imgB)
        lpips = self.lpips(imgA, imgB)
        #lr_psnr = self.psnr(imgA_lr, img_lr)
        mae = self.mae(imgA, imgB)
        mse = self.mse(imgA, imgB)        
        shift_mae = self.shift_l1_loss(imgA, imgB)
        res = {'psnr': psnr, 'ssim': ssim, 'lpips': lpips, 'mae': mae, 'mse': mse, "shift_mae": shift_mae}
        return {k: float(v) for k, v in res.items()}

    def lpips(self, imgA, imgB, model=None):
        device = next(self.model.parameters()).device
        tA = t(imgA).to(device)
        tB = t(imgB).to(device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):
        score, diff = ssim(imgA, imgB, full=True, channel_axis=-1, data_range=255)
        return score

    def psnr(self, imgA, imgB):
        return psnr(imgA, imgB, data_range=255)

    def mae(self, imgA, imgB):
        return np.abs(imgA - imgB).mean()

    def mse(self, imgA, imgB):
        return ((imgA.astype(np.int32) - imgB.astype(np.int32))**2).mean()

    def shift_l1_loss(self, imgA, imgB, border=3):
        """
        Modified mae to take into account pixel shifts
        """
        y_true = imgB.astype(np.int32)
        y_pred = imgA.astype(np.int32)
        max_pixels_shifts = 2*border
        size_image = y_true.shape[0]
        size_cropped_image = size_image - max_pixels_shifts
        patch_pred = y_pred[border:size_image -
                                    border, border:size_image-border,:]

        X = []
        for i in range(max_pixels_shifts+1):
            for j in range(max_pixels_shifts+1):
                patch_true = y_true[i:i+(size_image-max_pixels_shifts),
                                        j:j+(size_image-max_pixels_shifts),:]
                l1_loss =np.mean(np.abs(patch_true-patch_pred))
                X.append(l1_loss)

        min_l1 = min(X)

        return min_l1


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def im_crop(img_hr, img_lr, sr_scale):
    # TODO: clip for SRFlow
    c, h, w = img_hr.shape
    h = h - h % (sr_scale * 2)
    w = w - w % (sr_scale * 2)
    h_l = h // sr_scale
    w_l = w // sr_scale
    img_hr = img_hr[:, :h, :w]
    img_lr = img_lr[:, :h_l, :w_l]
    return img_hr, img_lr


def im_resize(batch, sr_factor):
    s = torch.tensor(batch[0][0].shape)*sr_factor
    img_lr_up=[]
    iter_ = iter(batch)
    if batch.shape[0]==1:
        s = np.squeeze(batch).shape[-1]
        im_up = cv.resize(np.array(np.transpose(np.squeeze(batch))), dsize=(s*sr_factor,s*sr_factor), interpolation=cv.INTER_CUBIC) #np.transpose(imresize(np.transpose(np.squeeze(batch)), sr_factor))
        return np.array(im_up)[None,:,:,:]
    
    for i in range(len(batch)):
        im_lr = next(iter_)
        s = im_lr.shape[-1]

        im_up = cv.resize(np.array(np.transpose(im_lr)), dsize=(s*sr_factor,s*sr_factor), interpolation=cv.INTER_CUBIC)
        img_lr_up.append(np.transpose(im_up))
    return np.array(img_lr_up)


def range_image(image):
    # image in [-1,1]
    return image*2-1


def multi_im_resize(batch, sr_factor):
    s = torch.tensor(batch[0][0].shape)*sr_factor
    img_lr_up=[]
    iter_ = iter(batch)
    if batch.shape[0]==1:
        s = np.squeeze(batch).shape[-1]
        im_up = cv.resize(np.array(np.transpose(np.squeeze(batch))), dsize=(s*sr_factor,s*sr_factor), interpolation=cv.INTER_CUBIC) #np.transpose(imresize(np.transpose(np.squeeze(batch)), sr_factor))
        return np.array(im_up)[None,:,:,:]
    
    for i in range(len(batch)):
        im_lr = next(iter_)
        s = im_lr.shape[-1]

        im_up = cv.resize(np.array(np.transpose(im_lr)), dsize=(s*sr_factor,s*sr_factor), interpolation=cv.INTER_CUBIC)
        img_lr_up.append(np.transpose(im_up))
    return np.array(img_lr_up)
