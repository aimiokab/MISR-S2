import torch.nn as nn
import torch
import numpy as np
import copy
from models.diffsr_modules import RRDBNet
from utils.hparams import hparams

import torch.nn.functional as F

from models.positional_encoding import PositionalEncoder

from .module_util import make_layer, initialize_weights
from .commons import Mish, SinusoidalPosEmb, RRDB, Residual, Rezero, LinearAttention
from .commons import ResnetBlock, Upsample, Block, Downsample
import functools
from models.HRNet import RecuversiveNet

class ResidualBlock(nn.Module):
    def __init__(self, channel_size=64, kernel_size=3):
        '''
        Args:
            channel_size : int, number of hidden channels
            kernel_size : int, shape of a 2D kernel
        '''
        
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )

    def forward(self, x):
        '''
        Args:
            x : tensor (B, C, W, H), hidden state
        Returns:
            x + residual: tensor (B, C, W, H), new hidden state
        '''
        
        residual = self.block(x)
        return x + residual


class HighResnetEncoder(nn.Module):
    def __init__(self, config):
        '''
        Args:
            config : dict, configuration file
        '''
        
        super(HighResnetEncoder, self).__init__()

        in_channels = config["in_channels"]
        num_layers = config["num_layers"]
        kernel_size = config["kernel_size"]
        channel_size = config["channel_size"]
        padding = kernel_size // 2

        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU())

        res_layers = [ResidualBlock(channel_size, kernel_size) for _ in range(num_layers)]
        self.res_layers = nn.Sequential(*res_layers)

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        '''
        Encodes an input tensor x.
        Args:
            x : tensor (B, C_in, W, H), input images
        Returns:
            out: tensor (B, C, W, H), hidden states
        '''
        
        x = self.init_layer(x)
        x = self.res_layers(x)
        x = self.final(x)
        return x
    



class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=64,
        n_head=16,
        d_k=4,
        mlp=[128, 64],
        dropout=0.2,
        d_model=128,
        T=1000,
        return_att=False,
        positional_encoding=True,
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = (
            out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        )  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w

        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, d_k
        )  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn
        

class Decoder(nn.Module):
    def __init__(self, config):
        '''
        Args:
            config : dict, configuration file
        '''
        
        super(Decoder, self).__init__()

        self.deconv = nn.Sequential(nn.ConvTranspose2d(in_channels=config["deconv"]["in_channels"],
                                                       out_channels=config["deconv"]["out_channels"],
                                                       kernel_size=config["deconv"]["kernel_size"],
                                                       stride=config["deconv"]["stride"]),
                                    nn.PReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(in_channels=config["deconv"]["out_channels"],
                                                       out_channels=config["deconv"]["out_channels"],
                                                       kernel_size=config["deconv"]["kernel_size"],
                                                       stride=config["deconv"]["stride"]),
                                    nn.PReLU())

        self.final = nn.Conv2d(in_channels=config["final"]["in_channels"],
                               out_channels=config["final"]["out_channels"],
                               kernel_size=config["final"]["kernel_size"],
                               padding=config["final"]["kernel_size"] // 2)

    def forward(self, x):
        '''
        Decodes a hidden state x.
        Args:
            x : tensor (B, C, W, H), hidden states
        Returns:
            out: tensor (B, C_out, 3*W, 3*H), fused hidden state
        '''
        
        x = self.deconv(x)
        x = self.deconv2(x)
        x = self.final(x)
        return x


class RRDB_MISR_Encoder(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super().__init__()
        hidden_size = hparams['hidden_size']
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, lrs, norm=True):

        batch_size, seq_len, c_in, heigth, width = lrs.shape

        if norm:
            lrs = (lrs + 1) / 2
        lrs = lrs.view(batch_size*seq_len, c_in, heigth, width)

        fea_first = fea = self.conv_first(lrs)
        for l in self.RRDB_trunk:
            fea = l(fea)
        trunk = self.trunk_conv(fea)
        fea = fea_first + trunk
        return fea


class RRDB_MISR_Decoder(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super().__init__()
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if hparams['sr_scale'] == 8:
            self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, fea, norm=True):

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if hparams['sr_scale'] == 8:
            fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea_hr = self.HRconv(fea)
        out = self.conv_last(self.lrelu(fea_hr))
        out = out.clamp(0, 1)
        if norm:
            out = out * 2 - 1

        return out


class HighResLtaeNet(nn.Module):
    def __init__(self, config):

        super(HighResLtaeNet, self).__init__()
        self.encoder = HighResnetEncoder(config["network"]["encoder"])
        self.temporal_encoder = LTAE2d()
        self.decoder = Decoder(config["network"]["decoder"])


    def forward(self, lrs, dates, config, get_fea=False):
        alphas = torch.tensor([0 for i in range(lrs.shape[1])])

        batch_size, seq_len, c_in, heigth, width = lrs.shape
        lrs = lrs.view(-1, seq_len, c_in, heigth, width)
        alphas = alphas.view(-1, seq_len, 1, 1, 1)

        if hparams["misr_ref_image"] == "median":
            refs, _ = torch.median(lrs[:, :9], 1, keepdim=True)  # reference image aka anchor, shared across multiple views
            refs = refs.repeat(1, seq_len, 1, 1, 1)
        if hparams["misr_ref_image"] == "closest":
            closest = torch.argmin(torch.abs(dates), 1)
            refs = lrs[0,closest[0],...][None,None,...]
            for e,c in enumerate(closest[1:]):
                refs = torch.cat([refs,lrs[e,c,...][None,None,...]])
            refs = refs.repeat(1, seq_len, 1, 1, 1)

        stacked_input = torch.cat([lrs, refs], 2) # tensor (B, L, 2*C_in, W, H)
        
        stacked_input = stacked_input.view(batch_size * seq_len, 2*c_in, width, heigth)

        fea = self.encoder(stacked_input)
        fea = fea.view(batch_size, seq_len, config['network']['encoder']['channel_size'], width, heigth)
        fea = self.temporal_encoder(fea, dates)
        out = self.decoder(fea)

        if get_fea:
            return out, fea
        else:
            return out


class RRDBLtaeNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, get_fea=False):
        super(RRDBLtaeNet, self).__init__()
        self.encoder = RRDB_MISR_Encoder(in_nc, out_nc, nf, nb, gc=32)
        self.temporal_encoder = LTAE2d(in_channels=32,
                                        n_head=16,
                                        d_k=4,
                                        mlp=[128, 32],
                                        dropout=0.2,
                                        d_model=128,
                                        T=1000,
                                        return_att=False,
                                        positional_encoding=True)
        self.decoder = RRDB_MISR_Decoder(in_nc, out_nc, nf, nb, gc=32)

    def forward(self, lrs, dates, config, get_fea=False):
        batch_size, seq_len, c_in, heigth, width = lrs.shape
        out = self.encoder(lrs)
        out = out.view(batch_size, seq_len, config['network']['encoder']['channel_size'], width, heigth)
        fea = self.temporal_encoder(out, dates)
        out = self.decoder(fea)
        if get_fea:
            return out, fea
        else:
            return out
    


class RRDBHighresNet(nn.Module):
    def __init__(self, config, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBHighresNet, self).__init__()
        self.encoder = RRDB_MISR_Encoder(in_nc, out_nc, nf, nb, gc=32)
        self.fuse = RecuversiveNet(config["recursive"])
        self.decoder = RRDB_MISR_Decoder(in_nc, out_nc, nf, nb, gc=32)

    def forward(self, lrs, alphas, config):
        batch_size, seq_len, c_in, heigth, width = lrs.shape
        alphas = alphas.view(-1, seq_len, 1, 1, 1)
        out = self.encoder(lrs, norm=False)
        out = out.view(batch_size, seq_len, config['network']['encoder']['channel_size'], width, heigth)
        fea = self.fuse(out, alphas)

        out = self.decoder(fea, norm=False)
        return out