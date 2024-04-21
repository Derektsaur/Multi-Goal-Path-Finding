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
class UNetwithRegression1(nn.Module):
    def __init__(self):
        super(UNetwithRegression1, self).__init__()


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

        reg1 = self.down1(x)
        reg2 = self.down2(reg1)
        reg3 = self.down3(reg2)
        reg4 = self.down4(reg3)
        reg5 = self.relu(self.fc1(torch.flatten(reg4,1,3)))
        # print(reg5.shape)
        reg = self.fc2(reg5)
        # print(reg.shape)
        return reg

if __name__ == '__main__':
    x=torch.randn(2,3,256,256)
    net=UNetwithRegression1()
    Seg, reg3, reg = net(x)
