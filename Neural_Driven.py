import torch
from torch import nn
from torch.nn import functional as f
from torchsummary import summary
import os


class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()
        self.h = h
        self.w = w
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='zeros', bias=False),
            # nn.BatchNorm2d(out_channel),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='zeros', bias=False),
            # nn.BatchNorm2d(out_channel),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True)  # modify
        )

    def forward(self, x):
        return self.layer(x)
    # 下采样 用卷积取代池化


class InveptionConv(nn.Module):  # Multi-scale convoltion module
    def __init__(self, inp, oup):  # 512,1024,512,256 -> 256，512，256，256
        super(InveptionConv, self).__init__()
        self.conv = Conv_Block(inp, inp * 2)
        self.conv1x1 = nn.Conv2d(inp * 2, inp, 1, 1, bias=False)
        self.dconv1 = nn.Conv2d(inp, inp, 3, 1, padding='same', dilation=2, padding_mode="zeros")
        self.dconv2 = nn.Conv2d(inp, inp, 3, 1, padding='same', dilation=4, padding_mode="zeros")
        self.con1x1o = nn.Conv2d(inp, inp, 1, 1, bias=False)
        self.se = SELayer(4 * inp)
        self.conv1x1oo = nn.Conv2d(4 * inp, oup, 1, 1, bias=False)

    def forward(self, x):
        x1 = self.conv1x1(self.conv(x))
        x2 = self.dconv1(x)
        x3 = self.dconv2(x)
        x4 = self.con1x1o(x)
        out = self.conv1x1oo(self.se(torch.concat((x1, x2, x3, x4), dim=1)))
        return out


class Deconv(nn.Module):
    def __init__(self, inp, oup, stride, output_padding):
        super(Deconv, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(inp, inp, 3, stride, 1, bias=False, output_padding=output_padding),  # groups=inp,
            # nn.BatchNorm2d(inp),
            nn.Dropout(0.3),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(inp, oup, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(oup),
            nn.Dropout(0.3),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):  # 下采样 用卷积取代池化
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            # nn.MaxPool2d(2,stride=2,padding=0),
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='zeros', bias=False),
            # nn.BatchNorm2d(channel),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        up = f.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


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


class block(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super(block, self).__init__()
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
            # nn.BatchNorm2d(hid_dim),
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


class Neural_Driven(nn.Module):
    def __init__(self):
        super(Neural_Driven, self).__init__()

        self.conv1_x = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        self.conv2_x = self._make_layer(3, 16, 24, stride=2)
        self.conv3_x = self._make_layer(4, 24, 40, stride=2)
        self.conv4_x = self._make_layer(6, 40, 80, stride=2)
        self.conv5_x = self._make_layer(3, 80, 256, stride=1)
        self.invep = InveptionConv(256, 256)
        self.dc1 = Deconv(280, 128, 1, output_padding=0)
        self.dc2 = Deconv(128, 64, 2, output_padding=1)
        self.dc3 = Deconv(64, 64, 2, output_padding=1)
        self.conv3x3 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out = nn.Conv2d(2, 1, 3, 1, 1)
        self.Th = nn.Sigmoid()
        for p in self.parameters():
            p.requires_grad = False
        self.c11 = Conv_Block(4, 64)
        self.d5 = DownSample(64)
        self.c12 = Conv_Block(64, 128)
        self.d6 = DownSample(128)
        self.c13 = Conv_Block(128, 256)
        self.u5 = UpSample(256)
        self.c14 = Conv_Block(256, 128)
        self.u6 = UpSample(128)
        self.c15 = Conv_Block(128, 64)
        self.c16 = nn.Conv2d(64, 2, 3, 1, 1)
        self.Th2 = nn.Sigmoid()
        self.out2 = nn.Conv2d(2, 1, 3, 1, 1)

    def forward(self, x):
        x = self.conv1_x(x)
        conv2_o = self.conv2_x(x)
        x = self.conv3_x(conv2_o)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.invep(x)
        x = f.interpolate(x, scale_factor=4, mode='nearest')
        x = self.dc1(torch.cat([conv2_o, x], dim=1))
        x = self.dc2(x)
        x = self.dc3(x)
        x = self.conv3x3(x)
        out1 = self.Th(self.out(x))

        C1 = self.c11(torch.concat([out1,x],dim=1))
        C2 = self.d5(C1)
        C3 = self.c12(C2)
        C4 = self.d6(C3)
        C5 = self.c13(C4)
        C6 = self.u5(C5, C3)
        C7 = self.c14(C6)
        C8 = self.u6(C7, C1)
        C9 = self.c15(C8)
        C10 = self.c16(C9)
        out2 = self.Th2(self.out2(C10))

        return out1, out2

    def _make_layer(self, n, inp, oup, stride):
        layers = []
        layers.append(block(inp, oup, stride))
        for i in range(n - 1):
            layers.append(block(oup, oup, stride=1))
        return nn.Sequential(*layers)

class Neural_Driven_pre(nn.Module):
    def __init__(self):
        super(Neural_Driven_pre, self).__init__()

        self.conv1_x = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        self.conv2_x = self._make_layer(3, 16, 24, stride=2)
        self.conv3_x = self._make_layer(4, 24, 40, stride=2)
        self.conv4_x = self._make_layer(6, 40, 80, stride=2)
        self.conv5_x = self._make_layer(3, 80, 256, stride=1)
        self.invep = InveptionConv(256, 256)
        self.dc1 = Deconv(280, 128, 1, output_padding=0)
        self.dc2 = Deconv(128, 64, 2, output_padding=1)
        self.dc3 = Deconv(64, 64, 2, output_padding=1)
        self.conv3x3 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out = nn.Conv2d(2, 1, 3, 1, 1)
        self.Th = nn.Sigmoid()

        self.c11 = Conv_Block(4, 64)
        self.d5 = DownSample(64)
        self.c12 = Conv_Block(64, 128)
        self.d6 = DownSample(128)
        self.c13 = Conv_Block(128, 256)
        self.u5 = UpSample(256)
        self.c14 = Conv_Block(256, 128)
        self.u6 = UpSample(128)
        self.c15 = Conv_Block(128, 64)
        self.c16 = nn.Conv2d(64, 2, 3, 1, 1)
        self.Th2 = nn.Sigmoid()
        self.out2 = nn.Conv2d(2, 1, 3, 1, 1)

    def forward(self, x):
        x = self.conv1_x(x)
        conv2_o = self.conv2_x(x)
        x = self.conv3_x(conv2_o)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.invep(x)
        x = f.interpolate(x, scale_factor=4, mode='nearest')
        x = self.dc1(torch.cat([conv2_o, x], dim=1))
        x = self.dc2(x)
        x = self.dc3(x)
        x = self.conv3x3(x)
        out1 = self.Th(self.out(x))

        C1 = self.c11(torch.concat([out1,x],dim=1))
        C2 = self.d5(C1)
        C3 = self.c12(C2)
        C4 = self.d6(C3)
        C5 = self.c13(C4)
        C6 = self.u5(C5, C3)
        C7 = self.c14(C6)
        C8 = self.u6(C7, C1)
        C9 = self.c15(C8)
        C10 = self.c16(C9)
        out2 = self.Th2(self.out2(C10))

        return out1, out2

    def _make_layer(self, n, inp, oup, stride):
        layers = []
        layers.append(block(inp, oup, stride))
        for i in range(n - 1):
            layers.append(block(oup, oup, stride=1))
        return nn.Sequential(*layers)

class Neural_Driven_post(nn.Module):
    def __init__(self):
        super(Neural_Driven_post, self).__init__()

        self.conv1_x = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        self.conv2_x = self._make_layer(3, 16, 24, stride=2)
        self.conv3_x = self._make_layer(4, 24, 40, stride=2)
        self.conv4_x = self._make_layer(6, 40, 80, stride=2)
        self.conv5_x = self._make_layer(3, 80, 256, stride=1)
        self.invep = InveptionConv(256, 256)
        self.dc1 = Deconv(280, 128, 1, output_padding=0)
        self.dc2 = Deconv(128, 64, 2, output_padding=1)
        self.dc3 = Deconv(64, 64, 2, output_padding=1)
        self.conv3x3 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out = nn.Conv2d(2, 1, 3, 1, 1)
        self.Th = nn.Sigmoid()

        self.c11 = Conv_Block(4, 64)
        self.d5 = DownSample(64)
        self.c12 = Conv_Block(64, 128)
        self.d6 = DownSample(128)
        self.c13 = Conv_Block(128, 256)
        self.u5 = UpSample(256)
        self.c14 = Conv_Block(256, 128)
        self.u6 = UpSample(128)
        self.c15 = Conv_Block(128, 64)
        self.c16 = nn.Conv2d(64, 2, 3, 1, 1)
        self.Th2 = nn.Sigmoid()
        self.out2 = nn.Conv2d(2, 1, 3, 1, 1)

    def forward(self, x):
        x = self.conv1_x(x)
        conv2_o = self.conv2_x(x)
        x = self.conv3_x(conv2_o)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.invep(x)
        x = f.interpolate(x, scale_factor=4, mode='nearest')
        x = self.dc1(torch.cat([conv2_o, x], dim=1))
        x = self.dc2(x)
        x = self.dc3(x)
        x = self.conv3x3(x)
        out1 = self.Th(self.out(x))

        C1 = self.c11(torch.concat([out1,x],dim=1))
        C2 = self.d5(C1)
        C3 = self.c12(C2)
        C4 = self.d6(C3)
        C5 = self.c13(C4)
        C6 = self.u5(C5, C3)
        C7 = self.c14(C6)
        C8 = self.u6(C7, C1)
        C9 = self.c15(C8)
        C10 = self.c16(C9)
        out2 = self.Th2(self.out2(C10))

        return out1, out2

    def _make_layer(self, n, inp, oup, stride):
        layers = []
        layers.append(block(inp, oup, stride))
        for i in range(n - 1):
            layers.append(block(oup, oup, stride=1))
        return nn.Sequential(*layers)

if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)

    net = Neural_Driven()
    out1, out2 = net(x)
    print(f"out1:{out1.shape}")
    print(f"out2:{out2.shape}")
    summary(net.to('cuda'), (3, 256, 256))

# filter(lambda p: p.requires_grad, model.parameters())
