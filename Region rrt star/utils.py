import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.nn import functional
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
def keep_image_size_open_fortest(path,i,size=(201,201)):
    openpath='path\%d.'
    openpath=os.path.join(path, '%d.jpg'%i)
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




def plot_figure(ave_Loss,epoch_num,title,x_label,y_label):

    # y = np.load(r'D:\Desktop\Unet RRT star\Unet-master-pytorch-v2/epoch_{}.npy'.format(n))
    y = ave_Loss
    # print(y)
    x=[]
    for i in range(epoch_num):
        x.append(i)
    plt.plot(x, y, '.-')
    plt_title = title
    plt.title(plt_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    f = plt.gcf()
    plt.show()
    f.savefig(F'FigureResult/'+title+'.jpg',dpi=300, bbox_inches='tight')

def draw_comparision(a,b,epoch,label1,label2,index,title):
    l1, = plt.plot(a,'-',color='#50A1DE')
    l2, = plt.plot(b,'-', color='#6E0000')

    plt.legend(handles=[l1,l2,], labels=[label1,label2], loc='best')
    plt.xlabel('epoch')
    plt.ylabel(index)
    plt.xlim(0,epoch)
    f = plt.gcf()
    plt.show()
    f.savefig(F'FigureResult/'+title+'.jpg',dpi=300, bbox_inches='tight')


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
    positive = np.sum(out_array >= 0.5) # TP+FN
    # print(positive)
    negative = np.sum(out_array < 0.5) # FP+TN
    true = np.sum(seg_array == 1) # TP+FP
    false = np.sum(seg_array == 0) # FN+TN
    tp = np.sum(unit_array >= 1.5) # TP
    tn = np.sum(unit_array < 0.5) # TN
    # print("positive:%d"%positive)
    if tp==0|positive==0:
        precision = 0
        recall = 0
        acc = 0
        fnr = 0.99  # false negative rate
        F1 = 0
        IOU = 0
        DICE = 0
    else:
        precision = tp/positive
        recall = tp/true
        acc = (tp + tn)/(positive + negative)
        fnr = (false - tn)/positive # false negative rate
        F1 = 2*precision*recall/(precision + recall)
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


