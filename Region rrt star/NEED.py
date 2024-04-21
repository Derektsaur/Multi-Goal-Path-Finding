import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

# from DepthwiseSeparableConvolution import depthwise_separable_conv as sep_conv

class separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, bias=False):
        super(separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin,
                                   bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class block(nn.Module):
    def __init__(self, in_channel, out_channel, identity_conv=None, stride=1):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.conv2 = separable_conv(out_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # self.conv3 = nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, stride=1, padding=0)
        self.conv3 = separable_conv(out_channel, out_channel * 4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channel * 4)
        self.relu = nn.ReLU()

        self.identity_conv = identity_conv

    def forward(self, x):
        identity = x.clone()  # 入力を保持する

        x = self.conv1(x)  # 1×1の畳み込み
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)  # 3×3の畳み込み（パターン3の時はstrideが2になるため、ここでsizeが半分になる）
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)  # 1×1の畳み込み
        x = self.bn3(x)

        if self.identity_conv is not None:
            identity = self.identity_conv(identity)
        x += identity

        x = self.relu(x)

        return x


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class Astrous(nn.Module):
    def __init__(self, channel, rate):
        super(Astrous, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, 256, kernel_size=3, padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class Deconv(nn.Module):
    def __init__(self, channel, stride, output_padding):
        super(Deconv, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(channel, channel, kernel_size=3, stride=stride, padding=1,
                               output_padding=output_padding),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class NEED(nn.Module):
    def __init__(self):
        super(NEED, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # stride 改 1
        self.conv2_x = self._make_layer( 3, res_block_in_channels=16, first_conv_out_channels=6, stride=2)
        self.conv3_x = self._make_layer( 4, res_block_in_channels=24, first_conv_out_channels=10, stride=2)
        self.conv4_x = self._make_layer( 6, res_block_in_channels=40, first_conv_out_channels=20, stride=2)
        self.conv5_x = self._make_layer( 3, res_block_in_channels=80, first_conv_out_channels=80, stride=1)

        self.aspp1 = Astrous(320, 1)
        self.aspp2 = Astrous(320, 2)
        self.aspp4 = Astrous(320, 4)
        self.aspp6 = Astrous(320, 6)
        self.a_conv = Conv_Block(320, 256)
        self.conv_1x1 = Conv_Block(1024, 256)

        # self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv_up = Conv_Block(256, 256)
        self.dc1 = Deconv(280, 1, output_padding=0)
        self.d1 = Conv_Block(280, 128)
        self.dc2 = Deconv(128, 2, output_padding=1)
        self.d2 = Conv_Block(128, 64)
        self.dc3 = Deconv(64, 2, output_padding=1)
        self.d3 = Conv_Block(64, 64)
        self.conv3x3 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out = nn.Conv2d(2, 1, 3, 1, 1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)  # in:(3,256*256)、out:(16,128*128)
        x = self.bn1(x)  # in:(16,128*128)、out:(16,128*128)
        x = self.relu(x)  # in:(16,128*128)、out:(16,128*128)
        x = self.maxpool(x)  # in:(16,128*128)、out:(16,128*128)
        conv2_o = self.conv2_x(x)  # in:(16,128*128)、out:(24,64*64)
        x = self.conv3_x(conv2_o)  # in:(24,64*64)、out:(40,32*32)
        x = self.conv4_x(x)  # in:(40,32*32)、out:(80,16*16)
        x = self.conv5_x(x)  # in:(80,16*16)、out:(320,16*16)

        a1 = (self.aspp1(x))
        a2 = (self.aspp2(x))
        a4 = (self.aspp4(x))
        a6 = (self.aspp6(x))
        x = self.conv_1x1(torch.cat([a1, a2, a4, a6], dim=1))  # (256,16x16)

        # x = self.upsample(x)  # (256,16x16) -> (256,64x64)
        x = F.interpolate(x, scale_factor=4, mode='nearest')
        x = self.conv_up(x)
        x = self.d1(self.dc1(torch.cat([conv2_o, x], dim=1)))  # (280,64x64)->[1, 128, 64, 64]
        x = self.d2(self.dc2(x))  # (128,64x64) -> [1, 64, 128, 128]
        x = self.d3(self.dc3(x))  # (64,128x128) -> [1, 64, 256, 256]
        x = self.conv3x3(x)  # (64,256x256) -> (2,256x256)
        x = self.Th(self.out(x))

        return x

    def _make_layer(self, num_res_blocks, res_block_in_channels, first_conv_out_channels, stride):
        layers = []
        # identity_conv = nn.Conv2d(res_block_in_channels, first_conv_out_channels * 4, kernel_size=1, stride=stride)
        identity_conv = separable_conv(res_block_in_channels, first_conv_out_channels * 4, kernel_size=1, stride=stride,
                                 padding=0)
        layers.append(block(res_block_in_channels, first_conv_out_channels, identity_conv, stride))
        in_channels = first_conv_out_channels * 4
        for i in range(num_res_blocks - 1):
            layers.append(block(in_channels, first_conv_out_channels, identity_conv=None, stride=1))

        return nn.Sequential(*layers)


if __name__ == '__main__':
    model = NEED()
    y = torch.rand([2, 3, 256, 256])
    print(f'input : {y.shape}')
    print(f'output : {model(y).shape}')
    # print(model.state_dict())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000
    print("Size of Model:{}".format(num_trainable_params))
