import math
import os

import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.nn import functional
from datetime import datetime
import logging
# 1 channels input segmentation for train

class Node:
    #定义Node类
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None  #通过逆向回溯路径
    #  x是纵向方向， y是横向， 左上角是零点

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
    openpath=os.path.join(path, '%d.png'%i)
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

# 输出为3




def plot_figure(ave_Loss,epoch_num,title,x_label,y_label):

    # y = np.load(r'D:\Desktop\Unet RRT star\Unet-master-pytorch-v2/epoch_{}.npy'.format(n))
    y = ave_Loss
    # print(y)
    x=[]
    for i in range(1, epoch_num+1):
        x.append(i)
    plt.plot(x, y, '.-')
    plt_title = title
    plt.title(plt_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    f = plt.gcf()
    plt.figure()
    # plt.show()
    f.savefig(r'figure_result/'+title+'.png',dpi=300, bbox_inches='tight')

def draw_comparision(a,b,epoch,label1,label2,index_x,index_y,title, log=False):
    epoch+=1
    l1, = plt.plot(a,'-',color='#50A1DE')
    l2, = plt.plot(b,'-', color='#6E0000')

    plt.legend(handles=[l1,l2,], labels=[label1,label2], loc='best')
    plt.xlabel(index_x)
    plt.ylabel(index_y)
    if log == True:
        plt.yscale('log')
    plt.xlim(-1,epoch)
    f = plt.gcf()
    plt.figure()
    # plt.show()
    f.savefig(r'figure_result/'+title+'.jpg',dpi=300, bbox_inches='tight')

def draw_comparision3d(ave_loss1, ave_loss2, ave_loss3, epoch,label1='seg1',label2='seg2',label3='reg',index_x='epoch',index_y='loss',title='difference loss for multi-task learning model'):
    l1, = plt.plot(ave_loss1, '-', color='#50A1DE')
    l2, = plt.plot(ave_loss2, '-', color='#6E0000')
    l3, = plt.plot(ave_loss3, '-', color='#1A0000')

    plt.legend(handles=[l1, l2, l3,], labels=[label1, label2, label3], loc='best')
    plt.xlabel(index_x)
    plt.ylabel(index_y)
    plt.xlim(0, epoch)
    f = plt.gcf()
    plt.show()
    f.savefig(r'C:\Users\huang\Desktop\S&Reg V2\TRAIN MODEL\figure_result/' + title + '.jpg', dpi=300, bbox_inches='tight')


def plot_loss_acc(a,b,epoch_num):
    l1, = plt.plot(a,'.-')
    l2, = plt.plot(b,'.-')
    plt.legend(handles=[l1, l2], labels=['loss', 'acc'], loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.xlim(0,epoch_num)
    plt.show()
    plt.savefig('acc_loss')
def evaluation(out,seg):
    unit = seg + out
    unit_array = unit.detach().cpu().numpy()
    out_array = out.detach().cpu().numpy()
    seg_array = seg.detach().cpu().numpy()
    # confusion matrix
    #       positive1  negative0
    # true1    TP       FP
    # false0   FN       TN
    positive = np.sum(out_array >= 0.5) # TP+FN？
    negative = np.sum(out_array < 0.5) # FP+TN？
    true = np.sum(seg_array == 1) # TP+FP？
    false = np.sum(seg_array == 0) # FN+TN？
    tp = np.sum(unit_array >= 1.5) # TP
    tn = np.sum(unit_array < 0.5) # TN

    eps=1e-9
    precision = tp/(positive + eps)
    # if tp*positive==0: precision=1e-4
    recall = tp/true
    acc = (tp + tn)/(positive + negative+eps)
    fnr = (false - tn)/(positive+eps) # false negative rate
    F1 = 2*precision*recall/(precision + recall+eps)
    IOU= tp/(true + false - tn)
    DICE = 2*tp/(true + positive)
    evaluation_criteria = [recall, precision, acc, fnr, F1, IOU, DICE]
    return evaluation_criteria

'''post-processing'''
def dilated_promising_region(out_array):

    ret, out_array = cv2.threshold(out_array, 0.5, 1, cv2.THRESH_BINARY)
    kernel = np.uint8(np.ones((5, 5)))
    out_array_dilate = cv2.dilate(out_array, kernel)
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
            if out_array_dilate[i][j]==1:
                promising_region_list.append(Node(i,j))
                promising_region_list_temp.append([i, j])
    return promising_region_list,promising_region_list_temp

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

def draw_promising_region(promising_region_list_temp,img,img_gray):
    # print(promising_region_list_temp)
    img=np.array(img)
    # print(img.shape[0])
    # print(img.shape[1])
    # print(img.shape[2])
    # print(promising_region_list_temp)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if [i,j] in promising_region_list_temp:
                if img_gray[i][j]!=0:
                # print([i,j])
                    img[i][j]=[0,0,110]
    return img

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):
        num = targets.size(0)
        smooth = 1

        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        dice = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)

        dice_loss = 1 - dice.sum() / num

        return dice_loss
def weightBCEloss(output, target, weight=None, pos_weight=None):
    # 处理正负样本不均衡问题
    if pos_weight is None:
        label_size = output.size()[1]
        pos_weight = torch.ones(label_size)
    # 处理多标签不平衡问题
    if weight is None:
        label_size = output.size()[1]
        weight = torch.ones(label_size)

    val = 0
    for li_x, li_y in zip(output, target):
        for i, xy in enumerate(zip(li_x, li_y)):
            x, y = xy
            loss_val = pos_weight[i] * y * math.log(x, math.e) + (1 - y) * math.log(1 - x, math.e)
            val += weight[i] * loss_val
    return torch.tensor(-val / (output.size()[0] * output.size(1)))

class Logger:

    def __init__(self, log_dir=None):
        if not log_dir:
            file_path = os.path.abspath(__file__)
            root_path = os.path.dirname(file_path)
            time = datetime.now()
            log_dir = os.path.join(root_path, time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S"))
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        logging.basicConfig(filename=log_dir + '/log.txt',
                            format='%(message)s',
                            level=logging.INFO,
                            filemode='a')

    def info(self, info, verbose=True):
        if verbose:
            print(info)
        logging.info(info)


if __name__ == '__main__':
    epoch = 10

    ave_loss1 = np.random.randn(epoch)
    ave_loss2 = np.random.randn(epoch)
    ave_loss3 = np.random.randn(epoch)
    draw_comparision3d(ave_loss1, ave_loss2, ave_loss3, epoch, label1='seg1', label2='seg2', label3='reg', index_x='epoch',
                   index_y='loss', title='difference loss for multi-task learning model')


