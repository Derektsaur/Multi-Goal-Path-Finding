import torch.nn as nn
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from SubLayers import MultiHeadAttention, PositionwiseFeedForward


class block(nn.Module):
    def __init__(self, inp, oup, stride=1, dropout=0.1):
        super(block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup
        hid_dim = inp * 6
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hid_dim, 1, 1, 0, bias=False),
            nn.Dropout(dropout),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hid_dim, hid_dim, 3, stride, 1, groups=hid_dim, bias=False),
            nn.Dropout(dropout),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hid_dim, oup, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Deconv(nn.Module):
    def __init__(self, inp, oup, stride, dropout=0.2):
        super(Deconv, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(inp, inp, 3, stride, 1, bias=False, output_padding=(stride - 1)),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(inp, oup, 1, 1, 0, bias=False),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel, dropout=0.2):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class SELayer(nn.Module):
    def __init__(self, c1, r=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // r, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)
        self.l2 = nn.Linear(c1 // r, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class InceptionConv(nn.Module):
    def __init__(self, inp, oup):
        super(InceptionConv, self).__init__()
        self.conv = Conv_Block(inp, inp * 2)
        self.conv1x1 = nn.Conv2d(inp * 2, inp, 1, 1, bias=False)
        self.dconv1 = nn.Conv2d(inp, inp, 3, 1, padding='same', dilation=2, padding_mode="zeros")
        self.dconv2 = nn.Conv2d(inp, inp, 3, 1, padding='same', dilation=4, padding_mode="zeros")
        self.con1x1o = nn.Conv2d(inp, inp, 1, 1, bias=False)
        self.se = SELayer(4 * inp)
        self.conv1x1oo = nn.Conv2d(4 * inp, oup, 1, 1, bias=False)
        self.conv2 = Conv_Block(4 * inp, inp)

    def forward(self, x):
        x1 = self.conv1x1(self.conv(x))
        x2 = self.dconv1(x)
        x3 = self.dconv2(x)
        x4 = self.con1x1o(x)
        out = self.conv1x1oo(self.se(torch.concat((x1, x2, x3, x4), dim=1)))
        return out


class UpSampling(nn.Module):
    def __init__(self, inp, oup, factor):
        super(UpSampling, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0),
            nn.Upsample(scale_factor=factor, mode='bilinear'),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output = torch.utils.checkpoint.checkpoint(self.slf_attn,
        enc_input, enc_input, enc_input, slf_attn_mask,use_reentrant=False)

        enc_output = self.pos_ffn(enc_output)
        return enc_output
