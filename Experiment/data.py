import torch
import torchvision.utils
from torch.utils.data import Dataset
import os
import numpy as np
import data
from utils import *
from torchvision import transforms
transform=transforms.Compose([transforms.ToTensor()])

class MyDataset(Dataset):

    def __init__(self,path):
        self.path=path
        self.name1=os.listdir(os.path.join(path,'label'))
        self.name2=os.listdir(os.path.join(path,'cost'))
        self.name3=os.listdir(os.path.join(path,'txt_coordinate'))
        self.name4=os.listdir(os.path.join(path,'line'))



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

        cost_name = self.name2[index]
        cost_path = os.path.join(self.path,'cost',cost_name)
        cost_file = open(cost_path,"r")
        cost = torch.tensor(float(cost_file.read(6)))

        coord_name = self.name3[index]
        coord_path = os.path.join(self.path,'txt_coordinate',coord_name)
        coord_file = open(coord_path,"r")
        coord = []

        lines = coord_file.readlines()  # 读取全部内容 ，并以列表方式返回
        for line in lines:
            coord.append(int(line.strip('\n')))
        # print(coord)
        return transform(image), transform(segment_image1), transform(segment_image2), cost, torch.Tensor(coord) #transfer to torch [0-1.0]

if __name__ == '__main__':
    data=MyDataset(r'C:\Users\huang\Desktop\[2]TSP Regression\Data_TSP')

    print(len(data))
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[0][2])

