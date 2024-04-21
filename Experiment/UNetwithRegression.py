import torch
from torch import nn
from torch.nn import functional as f
class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='zeros',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True)# modify
        )
    def forward(self,x):
        return self.layer(x)
    #下采样 用卷积取代池化

class Refine_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Refine_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True)  # modify
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            # nn.MaxPool2d(2,stride=2,padding=0),
            nn.Conv2d(channel,channel,3,2,1,padding_mode='zeros',bias=False),
            nn.BatchNorm2d(channel),
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
class UNetwithRegression(nn.Module):
    def __init__(self):
        super(UNetwithRegression, self).__init__()
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
        Seg = self.Th(self.out1(O5))
        reg1 = self.down1(Seg)
        reg2 = self.down2(reg1)
        reg3 = self.down3(reg2)
        reg4 = self.down4(reg3)

        reg5 = self.relu(self.fc1(torch.flatten(reg4,1,3)))
        # print(reg5.shape)
        reg = self.fc2(reg5)
        # print(reg.shape)
        return Seg,reg5,reg

if __name__ == '__main__':
    x=torch.randn(2,3,256,256)
    net=UNetwithRegression()
    print(net)
    Seg, reg3, reg = net(x)
