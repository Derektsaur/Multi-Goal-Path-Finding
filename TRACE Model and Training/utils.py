import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

from utils_2 import SoftDiceLoss


# 1 channels input segmentation for train

class Node:
    #定义Node类
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None  #通过逆向回溯路径
    #  x是纵向方向， y是横向， 左上角是零点

def draw_promising_region(promising_region_list_temp,img):
    # print(promising_region_list_temp)
    img=np.array(img)
    # print(img.shape[0])
    # print(img.shape[1])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if [i,j] in promising_region_list_temp:
                # print([i,j])
                img[i][j]=[0,0,110]
    # cv2.imshow('sssss',img)
    print('img out')
    # cv2.waitKey()
    return img



#获得promsing region的坐标
def keep_image_size_open(path,size=(256,256)):
    img=Image.open(path)
    temp=max(img.size)
    # print(temp)
    mask=Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    mask=mask.resize(size)
    mask=np.array(mask)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    return mask

# 3 channels input map for train
def keep_image_size_open_fortrain(path,size=(256,256)):
    img=Image.open(path)
    temp=max(img.size)
    # print(temp)
    mask=Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    mask=mask.resize(size)
    return mask
# 3 channel input image
def keep_image_size_open_fortest(path,i,size=(256,256)):
    openpath='path\%d.'
    #openpath=os.path.join(path, '%d.jpg'%i)
    openpath = os.path.join(path, '%d.png' % i)
    img=Image.open(openpath)
    temp=max(img.size)
    # print(temp)
    mask=Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    mask=mask.resize(size)
    return mask


def image_blend_out(alpha,i,epoch,save_path):
    img1=Image.open(F'{save_path}/img{i}_{epoch}.png')
    img2=Image.open(F'{save_path}/img_out{i}_{epoch}.png')
    img=Image.blend(img1,img2,alpha)
    Image.Image.save(img,F'{save_path}/results/{i}_{epoch}.png')

def image_blend_groundtruth(alpha,i,epoch,save_path):
    img1=Image.open(F'{save_path}/img{i}_{epoch}.png')
    img2=Image.open(F'{save_path}/img_true{i}_{epoch}.png')
    img=Image.blend(img1,img2,alpha)
    Image.Image.save(img,F'{save_path}/groundtruth/{i}_{epoch}.png')

def image_blend_test(alpha,i,save_path):

    img1=Image.open(F'{save_path}/{i}.png')
    # img2=keep_image_size_open(save_path,i)
    img2=Image.open(F'result/result_mix/{i}.png')
    img=Image.blend(img1,img2,alpha)
    Image.Image.save(img,F'result/result_mix/{i}.png')

def eight_adjacency_matrix(map, matrix, i, j):
    matrix = [[0 for k in range(9)] for k in range(9)]
    if map[i - 1][j - 1] == 1:
        matrix[0][0] == 1
    if map[i][j - 1] == 1:
        matrix[1][1] = 1
    if map[i + 1][j - 1] == 1:
        matrix[2][2] = 1
    if map[i - 1][j] == 1:
        matrix[3][3] = 1
    if map[i][j] == 1:
        matrix[4][4] = 1
    if map[i + 1][j] == 1:
        matrix[5][5] = 1
    if map[i - 1][j + 1] == 1:
        matrix[6][6] = 1
    if map[i][j + 1] == 1:
        matrix[7][7] = 1
    if map[i + 1][j + 1] == 1:
        matrix[8][8] = 1
    for m in range(9):
        for n in range(9):
            if matrix[m][m] * matrix[n][n] == 1:
                matrix[m][n] = 1
                matrix[n][m] = 1
    return matrix

def promising_node(map_binary,pos_list,adjacency_matrix_list):
    for i in range(1, 255):
        for j in range(1, 255):
            # print(map_binary)
            # print('*'*30)
            # print(map_binary[0])
            if map_binary[i][j] != 0:
                # print('*')
                matrix = [[0] * 3 * 3 for i in range(3 * 3)]
                matrix = eight_adjacency_matrix(map_binary, matrix, i, j)
                pos_list.append([i, j])
                adjacency_matrix_list.append(matrix)
    return pos_list,adjacency_matrix_list
# 输出为3
def adjacency_matrix_out(img):

    img=img.detach().cpu().numpy()
    # print(type(img))
    # print(np.size(img,0))
    # print(np.size(img,1))
    # print(img)
    # img=np.transpose(img,(2,0,1))
    map_binary = (img>0.6).astype(np.int_)[0][0]
    # print(map_binary)
    # ret, map_binary = cv2.threshold(map_gray, 127, 255, cv2.THRESH_BINARY)  # 二值图像0/255  512*512
    # p_max=np.where(np.max(img))
    # ret, map_binary = cv2.threshold(map_gray, 0.5, 1, cv2.THRESH_BINARY)

    adjacency_matrix_list = []
    pos_list = []
    pos_list,adjacency_matrix_list=promising_node(map_binary,pos_list,adjacency_matrix_list)
    return pos_list,adjacency_matrix_list


def adjacency_matrix(img):
    img=img.detach().cpu().numpy()
    # print(type(img))
    # print(type(img))
    # print(np.size(img,0))
    # print(np.size(img,1))
    # print(img)
    # img=np.transpose(img,(2,0,1))
    # map_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, map_binary = cv2.threshold(map_gray, 127, 255, cv2.THRESH_BINARY)  # 二值图像0/255  512*512
    # map_binary = map_binary / 255
    map_binary = img[0][0]
    adjacency_matrix_list = []
    pos_list = []
    pos_list,adjacency_matrix_list=promising_node(map_binary,pos_list,adjacency_matrix_list)
    return pos_list,adjacency_matrix_list

def calculate_adjacency_matrix_loss(pos_list,adjacency_matrix_list,pos_list_t,adjacency_matrix_list_t):
    calcu = nn.BCELoss()
    if pos_list:
        # print(pos_list)
        n1=len(pos_list)
        # print(n1)

        n2=len(pos_list_t[0])
        adjacency_matrix_array=torch.FloatTensor(adjacency_matrix_list)
        adjacency_matrix_array_t=torch.FloatTensor(adjacency_matrix_list_t)

        count = 0

        loss = torch.Tensor([0.0])

        falsematrix = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        truematrix = np.array([[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1]], dtype=np.float32)
        falsematrix = torch.from_numpy(falsematrix)
        truematrix = torch.from_numpy(truematrix)
        for i in range(n1):
            if pos_list[i] in pos_list_t:
                count +=1
                idx=pos_list_t.index(pos_list[i])

                loss += calcu(adjacency_matrix_array[i],adjacency_matrix_array_t[idx])
            else:
                loss += calcu(adjacency_matrix_array[i],falsematrix)
        for i in range(n2-count):
            loss += loss + calcu(truematrix,falsematrix)
    else:
        loss = torch.Tensor([0.0])
        falsematrix = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        # truematrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
        falsematrix = torch.from_numpy(falsematrix)
        # truematrix = torch.from_numpy(truematrix)

        n2 = len(pos_list_t)

        # adjacency_matrix_array = torch.FloatTensor(adjacency_matrix_list)
        adjacency_matrix_array_t = torch.FloatTensor(adjacency_matrix_list_t)
        for i in range(n2):
            loss += calcu(adjacency_matrix_array_t[i],falsematrix)

    return loss



def plot_loss(ave_Loss,epoch_num):

    # y = np.load(r'D:\Desktop\Unet RRT star\Unet-master-pytorch-v2/epoch_{}.npy'.format(n))
    y = ave_Loss
    print(y)
    # for i in range(0,n):
    #     enc = np.load(r'D:\Desktop\Unet RRT star\Unet-master-pytorch-v2/epoch_{}.npy'.format(i))
    #     # enc = torch.load('D:\MobileNet_v1\plan1-AddsingleLayer\loss\epoch_{}'.format(i))
    #     tempy = list(enc)
    #     y += tempy
    x=[]
    for i in range(epoch_num):
        x.append(i)
    plt.plot(x, y, '.-')
    plt_title = 'BATCH_SIZE = 2; LEARNING_RATE:0.001'
    plt.title(plt_title)
    plt.xlabel('Epoch')
    plt.ylabel('LOSS')
    # plt.savefig(file_name)
    plt.show()
def plot_loss_acc(a,b,epoch_num):
    l1, = plt.plot(a,'.-')
    l2, = plt.plot(b,'.-')
    plt.legend(handles=[l1, l2], labels=['loss', 'acc'], loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.xlim(0,epoch_num)
    plt.show()
    plt.savefig('acc_loss')
def calcu_acc(out,seg):
    acc=0
    ture_num=0
    for i in range(len(out)):
        unit=seg[i]+out[i]
        unit_array=unit.detach().cpu().numpy()
        seg_array=seg[i].detach().cpu().numpy()
        # print(type(seg))
        # print(unit_array)
        # print(np.shape(unit_array))
        # print(np.shape(seg_array))
        acc_num=np.sum(unit_array>255.5)
        # print(np.max(unit_array))
        # print(acc_num)
        total_true_num=np.sum(seg_array==1)
        # print(total_true_num)
        # print(acc_num/total_true_num)
        acc += acc_num
        ture_num+=total_true_num


    acc_val=acc/ture_num
    return acc_val

'''post-processing'''
def dilated_promising_region(out_array):

    ret, out_array = cv2.threshold(out_array, 0.6, 1, cv2.THRESH_BINARY)
    kernel = np.uint8(np.ones((5, 5)))
    out_array_dilate = cv2.erode(out_array, kernel)
    out_array_dilate = cv2.dilate(out_array_dilate, kernel)

    return out_array_dilate
def open_promising_region():
    1
def exstract_promising_region(out_array_dilate,img,promising_region_list):
    promising_region_list_temp=[]
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i][j]<=170:
    #             out_array_dilate[i][j]=0


    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if out_array_dilate[i][j]==1 and img[i][j] == 255:
                promising_region_list.append(Node(i,j))
                promising_region_list_temp.append([i, j])
    return promising_region_list_temp

def exstract_promising_region_with_obs(out_array_dilate,promising_region_list):
    promising_region_list_temp=[]
    for i in range(out_array_dilate.shape[0]):
        for j in range(out_array_dilate.shape[1]):
            if out_array_dilate[i][j] == 1:
                promising_region_list.append(Node(i,j))
                promising_region_list_temp.append([i,j])
    return promising_region_list,promising_region_list_temp




def cal_precision(out,seg,tp):


    out_array = out.detach().cpu().numpy()
    out_array = np.squeeze(out_array)
        # seg_array = seg[i].detach().cpu().numpy()
        # print(type(seg))
        # print(unit_array)
        # print(np.shape(unit_array))
        # print(np.shape(seg_array))

        # print(np.max(unit_array))
        # print(acc_num)
    p_predict = np.sum(out_array>0.5)

    precis = tp / p_predict
    return precis
def cal_recall(out,seg):

    out_array=out.detach().cpu().numpy()
    out_array=np.squeeze(out_array)
    seg_array=np.array(seg)/255
    unit_array=out_array+seg_array

    tp=np.sum(unit_array>1.5)

    p=np.sum(seg_array!=0)



    recall=tp/p
    return recall,tp
def cal_F1(out,seg):

    recall,tp=cal_recall(out,seg)
    precis=cal_precision(out,seg,tp)
    F1=2*precis*recall/(precis+recall)
    return F1




def jacobianloss(W1,reg1, lam):


    dh1 = reg1 * (1 - reg1)

    w_sum1 = torch.sum(Variable(W1) ** 2, dim=1)
    w_sum1 = w_sum1.unsqueeze(1)  # shape N_hidden x 1
    loss1 = torch.sum(torch.matmul (dh1 ** 2, w_sum1), 0)


    contractive_loss = loss1
    contractive_loss = contractive_loss.mean()
    return contractive_loss.mul_(lam)



def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为pillow
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    # assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # 转成pillow
    im = Image.fromarray(input_tensor).convert('RGB')
    im.save(filename, dpi=(300, 300), quality=95)

class MTLLoss(torch.nn.Module):
    def __init__(self, task_nums=2):
        super(MTLLoss, self).__init__()
        # x = torch.zeros([task_nums], dtype=torch.float, requires_grad=True)
        log_var2s = torch.ones([task_nums], dtype=torch.float, requires_grad=True)
        self.loss_fun1 = nn.BCELoss()
        self.loss_fun2 = SoftDiceLoss()
        self.log_var2s = nn.Parameter(log_var2s)


    def forward(self, logit_list, label_list, outimage1, segimage1, outimage2, segimage2):
        loss = 0
        mse = (logit_list - label_list) ** 2

        bcedice1 = self.loss_fun1(outimage1, segimage1) + self.loss_fun2(outimage1, segimage1)
        bcedice2 = self.loss_fun1(outimage2, segimage2) + self.loss_fun2(outimage2, segimage2)
        bcedice = (bcedice2 + bcedice1)/2
        pre0 = self.log_var2s[0] ** 2
        pre1 = self.log_var2s[1] ** 2

        loss += bcedice/(2*pre0) + mse/(2*pre1) + torch.log(1 + pre0) + torch.log(1 + pre1)

        return torch.mean(loss)
