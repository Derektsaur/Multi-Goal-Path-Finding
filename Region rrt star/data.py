import math
import random

import torch
import torchvision.utils
from torch.utils.data import Dataset
import os
from math import sqrt, pow, acos
import data
from utils import *
from torchvision import transforms

transform=transforms.Compose([transforms.ToTensor()])


class MyDataset(Dataset):

    def __init__(self,path):
        self.path=path
        self.name1=os.listdir(os.path.join(path,'label'))
        self.name2=os.listdir(os.path.join(path,'line'))
    def __len__(self):
        return len(self.name1)

    def __getitem__(self, index):
        segment_name=self.name1[index]  #0.png
        segment_path1=os.path.join(self.path,'label',segment_name)
        segment_path2=os.path.join(self.path,'line',segment_name)
        image_path=os.path.join(self.path,'train',segment_name)

        segment_image1=keep_image_size_open(segment_path1)
        segment_image2=keep_image_size_open(segment_path2)
        image=keep_image_size_open_fortrain(image_path)

        return transform(image), transform(segment_image1), transform(segment_image2)
if __name__ == '__main__':
    data=MyDataset('data')
    # print(data[0][0].shape)
    # print(data[0][1].shape)
    print(data[0][2])


