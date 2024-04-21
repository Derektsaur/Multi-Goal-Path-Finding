'''Define the Layers
Derived from - https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Layers.py
'''
import numpy as np
import torch.utils.checkpoint
import torch.nn as nn
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from einops import rearrange
from SubLayers import MultiHeadAttention, PositionwiseFeedForward


class Inverted_Block(nn.Module):  # InvertedResidual block (channel change)
    def __init__(self, inp, oup, stride=1):
        super(Inverted_Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup
        hid_dim = inp * 6  # expand_ratio = 6
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hid_dim, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(hid_dim),
            nn.Dropout(0.1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hid_dim, hid_dim, 3, stride, 1, groups=hid_dim, bias=False),
            nn.BatchNorm2d(hid_dim),
            nn.Dropout(0.1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hid_dim, oup, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Conv_Block(nn.Module):  # Channel change
    def __init__(self, in_channel, out_channel, dropout=0.1):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1, groups=in_channel, padding_mode='zeros', bias=False),
            # nn.BatchNorm2d(out_channel),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, padding_mode='zeros', bias=False),
            # nn.BatchNorm2d(out_channel),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True)  # modify
        )

    def forward(self, x):
        return self.layer(x)


class UpSampling(nn.Module):
    def __init__(self, inp, oup):
        super(UpSampling, self).__init__()
        self.layer = nn.Sequential(
            # nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(inp, oup, 3, 1, 1),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class DownSampling(nn.Module):
    def __init__(self, inp, oup, stride):
        super(DownSampling, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class Deconv(nn.Module):
    def __init__(self, inp, oup, stride):
        super(Deconv, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(inp, oup, 3, stride, 1, bias=False, output_padding=(stride - 1)),
            nn.Dropout(0.1),
            # nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # nn.ConvTranspose2d(inp, oup, 1, 1, 0, bias=False),
            # nn.Dropout(0.1),
            # nn.BatchNorm2d(oup),
            # nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class ResBlock(nn.Module):  # used in decoder
    def __init__(self, inp, c):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(inp, c, 1, 1, 0, bias=False)
        self.res = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.Dropout(0.1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.Dropout(0.1),
            nn.LeakyReLU(inplace=True),
            # nn.Conv2d(c, c, 1, 1, 0, bias=False),
        )

    def forward(self, x, feature_map=None):
        if feature_map is None:
            return x + self.res(x)
        else:
            x = self.conv(torch.cat([x, feature_map], dim=1))
            return x + self.res(x)


class block(nn.Module):  # InvertedResidual block (channel change)
    def __init__(self, inp, oup, stride=1, dropout=0.1):
        super(block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup
        hid_dim = inp * 6  # expand_ratio = 6
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hid_dim, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(hid_dim),
            nn.Dropout(dropout),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hid_dim, hid_dim, 3, stride, 1, groups=hid_dim, bias=False),
            # nn.BatchNorm2d(hid_dim),
            nn.Dropout(dropout),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hid_dim, oup, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EncoderLayer(nn.Module):
    ''' Single Encoder layer, that consists of a MHA layers and positiion-wise
    feedforward layer.
    '''

    def __init__(self, d_model, n_head, dropout=0.1):
        '''
        Initialize the module.
        :param d_model: Dimension of input/output of this layer
        :param d_inner: Dimension of the hidden layer of hte position-wise feedforward layer
        :param n_head: Number of self-attention modules
        :param d_k: Dimension of each Key
        :param d_v: Dimension of each Value
        :param dropout: Argument to the dropout layer.
        '''
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        '''
        The forward module:
        :param enc_input: The input to the encoder.
        :param slf_attn_mask: TODO ......
        '''
        # # Without gradient Checking
        # enc_output = self.slf_attn(
        #     enc_input, enc_input, enc_input, mask=slf_attn_mask)

        # With Gradient Checking !use_reentrant=False
        enc_output = torch.utils.checkpoint.checkpoint(self.slf_attn,
                                                       enc_input, enc_input, enc_input, slf_attn_mask,
                                                       use_reentrant=False)

        # enc_output, enc_slf_attn = self.slf_attn(
        #     enc_input, enc_input, enc_input, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)
        return enc_output


if __name__ == '__main__':
    model = EncoderLayer(512)
    x = torch.rand([1, 512, 16, 16])
    out = model(x)
    print(x.shape)
    print(out.shape)
