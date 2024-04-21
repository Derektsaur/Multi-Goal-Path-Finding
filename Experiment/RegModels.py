import torch
from torch import nn
from torch.nn import functional as f
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
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='zeros',bias=False),
            # nn.BatchNorm2d(out_channel),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='zeros', bias=False),
            # nn.BatchNorm2d(out_channel),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True)# modify
        )
    def forward(self,x):
        return self.layer(x)
    #下采样 用卷积取代池化

class InveptionConv(nn.Module):
    def __init__(self,in_channel,out_channel): #512,1024,512,256 --》256，512，256，256
        super(InveptionConv, self).__init__()
        self.conv = Conv_Block(in_channel,out_channel)
        self.se = SELayer(out_channel)
        self.conv1x1 = nn.Conv2d(out_channel, in_channel, 1, 1, bias=False)
        self.dconv1 = nn.Conv2d(in_channel,in_channel,3,1,padding='same',dilation = 2,padding_mode="zeros")
        self.dconv2 = nn.Conv2d(in_channel,in_channel,3,1,padding='same',dilation = 4,padding_mode="zeros")
        self.con1x1o = nn.Conv2d(in_channel, in_channel, 1, 1, bias=False)
        self.conv1x1oo = nn.Conv2d(2*out_channel, out_channel, 1, 1,bias=False)

    def forward(self,x):
        x1 = self.conv1x1(self.se(self.conv(x)))
        x2 = self.dconv1(x)
        x3 = self.dconv2(x)
        x4 = self.con1x1o(x)
        out = self.conv1x1oo(torch.concat((x1,x2,x3,x4),dim=1))
        return out
    #下采样 用卷积取代池化

class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            # nn.MaxPool2d(2,stride=2,padding=0),
            nn.Conv2d(channel,channel,3,2,1,padding_mode='zeros',bias=False),
            # nn.BatchNorm2d(channel),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)

    def forward(self,x,feature_map ):
        up=f.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)


'''SE layer'''
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

''''''

''''''
class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.se1 = SELayer(1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.se2 = SELayer(512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.se3 = SELayer(256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.se4 = SELayer(128)
        self.u4 = UpSample(128)
        # self.se5=SELayer(128)
        self.c9 = Conv_Block(128, 64)
        # 修改输出为0-1二值地图
        self.c10 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out1 = nn.Conv2d(2, 1, 3, 1, 1)
        self.Th = nn.Sigmoid()

        self.c11 = Conv_Block(1, 64)
        self.d5 = DownSample(64)
        self.c12 = Conv_Block(64, 128)
        self.d6 = DownSample(128)
        self.c13 = Conv_Block(128, 256)
        self.u5 = UpSample(256)
        self.c14 = Conv_Block(256, 128)
        self.u6 = UpSample(128)
        self.c15 = Conv_Block(128, 64)
        self.c16 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out2 = nn.Conv2d(2, 1, 3, 1, 1)

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.se1(self.c5(self.d4(R4)))
        O1 = self.se2(self.c6(self.u1(R5, R4)))
        O2 = self.se3(self.c7(self.u2(O1, R3)))
        O3 = self.se4(self.c8(self.u3(O2, R2)))
        O4 = self.c9(self.u4(O3, R1))
        # 增加1x3x3卷积层
        O5 = self.c10(O4)

        out1 = self.Th(self.out1(O5))

        C1 = self.c11(out1)
        C2 = self.d5(C1)
        C3 = self.c12(C2)
        C4 = self.d6(C3)
        C5 = self.c13(C4)
        C6 = self.u5(C5, C3)
        C7 = self.c14(C6)
        C8 = self.u6(C7, C1)
        C9 = self.c15(C8)
        C10 = self.c16(C9)
        out2 = self.Th(self.out2(C10))
        return out1, out2
class SRegNet_pre(nn.Module):
    def __init__(self):
        super(SRegNet_pre, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.se1 = SELayer(1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.se2 = SELayer(512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.se3 = SELayer(256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.se4 = SELayer(128)
        self.u4 = UpSample(128)
        # self.se5=SELayer(128)
        self.c9 = Conv_Block(128, 64)
        # 修改输出为0-1二值地图
        self.c10 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out1 = nn.Conv2d(2, 1, 3, 1, 1)
        self.Th = nn.Sigmoid()


        self.c11 = Conv_Block(1,64)
        self.d5 = DownSample(64)
        self.c12 = Conv_Block(64,128)
        self.d6 = DownSample(128)
        self.c13 = Conv_Block(128,256)
        self.u5 = UpSample(256)
        self.c14 = Conv_Block(256,128)
        self.u6 = UpSample(128)
        self.c15 = Conv_Block(128,64)
        self.c16 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out2 = nn.Conv2d(2, 1, 3, 1, 1)


        self.down1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 128
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )#64
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 32
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 16

        self.fc1 = nn.Linear(16*16*256,1024)
        self.fc2 = nn.Linear(1024,1)

        self.relu = nn.ReLU()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.se1(self.c5(self.d4(R4)))
        O1 = self.se2(self.c6(self.u1(R5, R4)))
        O2 = self.se3(self.c7(self.u2(O1, R3)))
        O3 = self.se4(self.c8(self.u3(O2, R2)))
        O4 = self.c9(self.u4(O3, R1))
        # 增加1x3x3卷积层
        O5 = self.c10(O4)


        out1 = self.Th(self.out1(O5))

        C1 = self.c11(out1)
        C2 = self.d5(C1)
        C3 = self.c12(C2)
        C4 = self.d6(C3)
        C5 = self.c13(C4)
        C6 = self.u5(C5,C3)
        C7 = self.c14(C6)
        C8 = self.u6(C7,C1)
        C9 = self.c15(C8)
        C10 = self.c16(C9)
        out2 = self.Th(self.out2(C10))

        reg1 = self.down1(out2)
        reg2 = self.down2(reg1)
        reg3 = self.down3(reg2)
        reg4 = self.down4(reg3)
        reg5 = self.relu(self.fc1(torch.flatten(reg4,1,3)))
        # print(reg5.shape)
        reg = self.fc2(reg5)
        # print(reg.shape)
        return out1,out2,reg

class SRegNet1(nn.Module):
    def __init__(self):
        super(SRegNet1, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.se1 = SELayer(1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.se2 = SELayer(512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.se3 = SELayer(256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.se4 = SELayer(128)
        self.u4 = UpSample(128)
        # self.se5=SELayer(128)
        self.c9 = Conv_Block(128, 64)
        # 修改输出为0-1二值地图
        self.c10 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out1 = nn.Conv2d(2, 1, 3, 1, 1)
        self.Th = nn.Sigmoid()

        self.c11 = Conv_Block(1, 64)
        self.d5 = DownSample(64)
        self.c12 = Conv_Block(64, 128)
        self.d6 = DownSample(128)
        self.c13 = Conv_Block(128, 256)
        self.u5 = UpSample(256)
        self.c14 = Conv_Block(256, 128)
        self.u6 = UpSample(128)
        self.c15 = Conv_Block(128, 64)
        self.c16 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out2 = nn.Conv2d(2, 1, 3, 1, 1)

        self.down1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 128
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )#64
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 32
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 16

        self.fc1 = nn.Linear(16*16*256,1024)
        self.fc2 = nn.Linear(1024,1)

        self.relu = nn.ReLU()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.se1(self.c5(self.d4(R4)))
        O1 = self.se2(self.c6(self.u1(R5, R4)))
        O2 = self.se3(self.c7(self.u2(O1, R3)))
        O3 = self.se4(self.c8(self.u3(O2, R2)))
        O4 = self.c9(self.u4(O3, R1))
        # 增加1x3x3卷积层
        O5 = self.c10(O4)
        out1 = self.Th(self.out1(O5))

        C1 = self.c11(out1)
        C2 = self.d5(C1)
        C3 = self.c12(C2)
        C4 = self.d6(C3)
        C5 = self.c13(C4)
        C6 = self.u5(C5, C3)
        C7 = self.c14(C6)
        C8 = self.u6(C7, C1)
        C9 = self.c15(C8)
        C10 = self.c16(C9)
        out2 = self.Th(self.out2(C10))

        reg1 = self.down1(out2)
        reg2 = self.down2(reg1)
        reg3 = self.down3(reg2)
        reg4 = self.down4(reg3)
        reg5 = self.relu(self.fc1(torch.flatten(reg4,1,3)))
        # print(reg5.shape)
        reg = self.fc2(reg5)
        # print(reg.shape)
        return out1, out2, reg

class SRegNet2(nn.Module):
    def __init__(self):
        super(SRegNet2, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.se1 = SELayer(1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.se2 = SELayer(512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.se3 = SELayer(256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.se4 = SELayer(128)
        self.u4 = UpSample(128)
        # self.se5=SELayer(128)
        self.c9 = Conv_Block(128, 64)
        # 修改输出为0-1二值地图
        self.c10 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out1 = nn.Conv2d(2, 1, 3, 1, 1)
        self.Th1 = nn.Sigmoid()

        for p in self.parameters():
            p.requires_grad = False

        self.Th2 = nn.Sigmoid()

        self.c11 = Conv_Block(1, 64)
        self.d5 = DownSample(64)
        self.c12 = Conv_Block(64, 128)
        self.d6 = DownSample(128)
        self.c13 = Conv_Block(128, 256)
        self.u5 = UpSample(256)
        self.c14 = Conv_Block(256, 128)
        self.u6 = UpSample(128)
        self.c15 = Conv_Block(128, 64)
        self.c16 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out2 = nn.Conv2d(2, 1, 3, 1, 1)

        self.down1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 128
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )#64
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 32
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 16

        self.fc1 = nn.Linear(16*16*256,1024)
        self.fc2 = nn.Linear(1024,1)

        self.relu = nn.ReLU()

    def forward(self, x, y):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.se1(self.c5(self.d4(R4)))
        O1 = self.se2(self.c6(self.u1(R5, R4)))
        O2 = self.se3(self.c7(self.u2(O1, R3)))
        O3 = self.se4(self.c8(self.u3(O2, R2)))
        O4 = self.c9(self.u4(O3, R1))
        # 增加1x3x3卷积层
        O5 = self.c10(O4)
        out1 = self.Th1(self.out1(O5))

        C1 = self.c11(y)
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

        reg1 = self.down1(out2)
        reg2 = self.down2(reg1)
        reg3 = self.down3(reg2)
        reg4 = self.down4(reg3)
        reg5 = self.relu(self.fc1(torch.flatten(reg4, 1, 3)))
        # print(reg5.shape)
        reg = self.fc2(reg5)
        # print(reg.shape)
        return out1, out2, reg

class SRegNet3(nn.Module):
    def __init__(self):
        super(SRegNet3, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.se1 = SELayer(1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.se2 = SELayer(512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.se3 = SELayer(256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.se4 = SELayer(128)
        self.u4 = UpSample(128)
        # self.se5=SELayer(128)
        self.c9 = Conv_Block(128, 64)
        # 修改输出为0-1二值地图
        self.c10 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out1 = nn.Conv2d(2, 1, 3, 1, 1)
        self.Th1 = nn.Sigmoid()

        self.c11 = Conv_Block(1, 64)
        self.d5 = DownSample(64)
        self.c12 = Conv_Block(64, 128)
        self.d6 = DownSample(128)
        self.c13 = Conv_Block(128, 256)
        self.u5 = UpSample(256)
        self.c14 = Conv_Block(256, 128)
        self.u6 = UpSample(128)
        self.c15 = Conv_Block(128, 64)
        self.c16 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out2 = nn.Conv2d(2, 1, 3, 1, 1)
        self.Th2 = nn.Sigmoid()

        for p in self.parameters():
            p.requires_grad = False

        self.down1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 128
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 64
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 32
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 16

        self.fc1 = nn.Linear(16 * 16 * 256, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.se1(self.c5(self.d4(R4)))
        O1 = self.se2(self.c6(self.u1(R5, R4)))
        O2 = self.se3(self.c7(self.u2(O1, R3)))
        O3 = self.se4(self.c8(self.u3(O2, R2)))
        O4 = self.c9(self.u4(O3, R1))
        # 增加1x3x3卷积层
        O5 = self.c10(O4)
        out1 = self.Th1(self.out1(O5))

        C1 = self.c11(out1)
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

        reg1 = self.down1(out2)
        reg2 = self.down2(reg1)
        reg3 = self.down3(reg2)
        reg4 = self.down4(reg3)
        reg5 = self.relu(self.fc1(torch.flatten(reg4, 1, 3)))
        # print(reg5.shape)
        reg = self.fc2(reg5)
        # print(reg.shape)
        return out1, out2, reg


class SRegNet_mod(nn.Module):
    def __init__(self):
        super(SRegNet_mod1, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = InveptionConv(512, 1024)
        self.se1 = SELayer(1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.se2 = SELayer(512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.se3 = SELayer(256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.se4 = SELayer(128)
        self.u4 = UpSample(128)
        # self.se5=SELayer(128)
        self.c9 = Conv_Block(128, 64)
        # 修改输出为0-1二值地图
        self.c10 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out1 = nn.Conv2d(2, 1, 3, 1, 1)
        self.Th1 = nn.Sigmoid()

        self.c11 = Conv_Block(1, 64)
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

        self.down1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 128
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 64
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 32
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 16
        self.down5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 8
        self.down6 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 4

        self.fc1 = nn.Linear(4 * 4 * 1024, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.se1(self.c5(self.d4(R4)))
        O1 = self.se2(self.c6(self.u1(R5, R4)))
        O2 = self.se3(self.c7(self.u2(O1, R3)))
        O3 = self.se4(self.c8(self.u3(O2, R2)))
        O4 = self.c9(self.u4(O3, R1))
        # 增加1x3x3卷积层
        O5 = self.c10(O4)

        out1 = self.Th1(self.out1(O5))

        C1 = self.c11(out1)
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

        out = torch.concat((out1, out2), dim=1)
        reg1 = self.down1(out)
        reg2 = self.down2(reg1)
        reg3 = self.down3(reg2)
        reg4 = self.down4(reg3)
        reg5 = self.down5(reg4)
        reg6 = self.down6(reg5)

        reg7 = self.relu(self.fc1(torch.flatten(reg6, 1, 3)))
        reg = self.fc2(reg7)
        # reg5 = self.relu(self.fc1(torch.flatten(reg4, 1, 3)))
        # print(reg5.shape)
        # reg = self.fc2(reg5)

        return out1, out2, reg
class SRegNet_mod1(nn.Module):
    def __init__(self):
        super(SRegNet_mod1, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = InveptionConv(512, 1024)
        self.se1 = SELayer(1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.se2 = SELayer(512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.se3 = SELayer(256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.se4 = SELayer(128)
        self.u4 = UpSample(128)
        # self.se5=SELayer(128)
        self.c9 = Conv_Block(128, 64)
        # 修改输出为0-1二值地图
        self.c10 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out1 = nn.Conv2d(2, 1, 3, 1, 1)
        self.Th1 = nn.Sigmoid()

        self.c11 = Conv_Block(1, 64)
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

        self.down1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 128
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 64
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 32
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 16
        self.down5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 8
        self.down6 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 4

        self.fc1 = nn.Linear(4 * 4 * 1024, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.se1(self.c5(self.d4(R4)))
        O1 = self.se2(self.c6(self.u1(R5, R4)))
        O2 = self.se3(self.c7(self.u2(O1, R3)))
        O3 = self.se4(self.c8(self.u3(O2, R2)))
        O4 = self.c9(self.u4(O3, R1))
        # 增加1x3x3卷积层
        O5 = self.c10(O4)

        out1 = self.Th1(self.out1(O5))

        C1 = self.c11(out1)
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

        out = torch.concat((out1, out2), dim=1)
        reg1 = self.down1(out)
        reg2 = self.down2(reg1)
        reg3 = self.down3(reg2)
        reg4 = self.down4(reg3)
        reg5 = self.down5(reg4)
        reg6 = self.down6(reg5)

        reg7 = self.relu(self.fc1(torch.flatten(reg6, 1, 3)))
        reg = self.fc2(reg7)
        # reg5 = self.relu(self.fc1(torch.flatten(reg4, 1, 3)))
        # print(reg5.shape)
        # reg = self.fc2(reg5)

        return out1, out2, reg
class SRegNet_mod2(nn.Module):
    def __init__(self):
        super(SRegNet_mod2, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = InveptionConv(512, 1024)
        self.se1 = SELayer(1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.se2 = SELayer(512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.se3 = SELayer(256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.se4 = SELayer(128)
        self.u4 = UpSample(128)
        # self.se5=SELayer(128)
        self.c9 = Conv_Block(128, 64)
        # 修改输出为0-1二值地图
        self.c10 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out1 = nn.Conv2d(2, 1, 3, 1, 1)
        self.Th1 = nn.Sigmoid()
        for p in self.parameters():
            p.requires_grad = False
        self.c11 = Conv_Block(1, 64)
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

        self.down1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 128
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 64
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 32
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 16
        self.down5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 8
        self.down6 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 4

        self.fc1 = nn.Linear(4 * 4 * 1024, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.se1(self.c5(self.d4(R4)))
        O1 = self.se2(self.c6(self.u1(R5, R4)))
        O2 = self.se3(self.c7(self.u2(O1, R3)))
        O3 = self.se4(self.c8(self.u3(O2, R2)))
        O4 = self.c9(self.u4(O3, R1))
        # 增加1x3x3卷积层
        O5 = self.c10(O4)

        out1 = self.Th1(self.out1(O5))

        C1 = self.c11(out1)
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

        out = torch.concat((out1, out2), dim=1)
        reg1 = self.down1(out)
        reg2 = self.down2(reg1)
        reg3 = self.down3(reg2)
        reg4 = self.down4(reg3)
        reg5 = self.down5(reg4)
        reg6 = self.down6(reg5)

        reg7 = self.relu(self.fc1(torch.flatten(reg6, 1, 3)))
        reg = self.fc2(reg7)
        # reg5 = self.relu(self.fc1(torch.flatten(reg4, 1, 3)))
        # print(reg5.shape)
        # reg = self.fc2(reg5)

        return out1, out2, reg
class SRegNet_mod3(nn.Module):
    def __init__(self):
        super(SRegNet_mod3, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = InveptionConv(512, 1024)
        self.se1 = SELayer(1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.se2 = SELayer(512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.se3 = SELayer(256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.se4 = SELayer(128)
        self.u4 = UpSample(128)
        # self.se5=SELayer(128)
        self.c9 = Conv_Block(128, 64)
        # 修改输出为0-1二值地图
        self.c10 = nn.Conv2d(64, 2, 3, 1, 1)
        self.out1 = nn.Conv2d(2, 1, 3, 1, 1)
        self.Th1 = nn.Sigmoid()

        self.c11 = Conv_Block(1, 64)
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
        for p in self.parameters():
            p.requires_grad = False
        self.down1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 128
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 64
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 32
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 16
        self.down5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 8
        self.down6 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0)
        )  # 4

        self.fc1 = nn.Linear(4 * 4 * 1024, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()

    def forward(self,x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.se1(self.c5(self.d4(R4)))
        O1 = self.se2(self.c6(self.u1(R5, R4)))
        O2 = self.se3(self.c7(self.u2(O1, R3)))
        O3 = self.se4(self.c8(self.u3(O2, R2)))
        O4 = self.c9(self.u4(O3, R1))
        # 增加1x3x3卷积层
        O5 = self.c10(O4)

        out1 = self.Th1(self.out1(O5))

        C1 = self.c11(out1)
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

        out = torch.concat((out1, out2), dim=1)
        reg1 = self.down1(out)
        reg2 = self.down2(reg1)
        reg3 = self.down3(reg2)
        reg4 = self.down4(reg3)
        reg5 = self.down5(reg4)
        reg6 = self.down6(reg5)

        reg7 = self.relu(self.fc1(torch.flatten(reg6, 1, 3)))
        reg = self.fc2(reg7)
        # reg5 = self.relu(self.fc1(torch.flatten(reg4, 1, 3)))
        # print(reg5.shape)
        # reg = self.fc2(reg5)

        return out1, out2, reg
if __name__ == '__main__':
    x=torch.randn(2,3,256,256)

    net2 = SRegNet_mod()
    out1, out2, reg = net2(x)
    print(net2)


# filter(lambda p: p.requires_grad, model.parameters())