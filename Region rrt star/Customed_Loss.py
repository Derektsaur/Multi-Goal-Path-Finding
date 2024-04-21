

from utils import *

from torch import nn,optim

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from math import exp

# reverse ：求纯度
#original： 求杂度 越杂越重要
class FocalLoss(nn.Module):
    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:  # alpha 是平衡因子
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma  # 指数
        self.class_num = class_num  # 类别数目
        self.size_average = size_average  # 返回的loss是否需要mean一下

    def forward(self, inputs, targets):
        # target : N, 1, H, W
        inputs = inputs.permute(0, 2, 3, 1)
        targets = targets.permute(0, 2, 3, 1)
        num, h, w, C = inputs.size()
        N = num * h * w
        inputs = inputs.reshape(N, -1)   # N, C
        targets = targets.reshape(N, -1)  # 待转换为one hot label
        P = F.softmax(inputs, dim=1)  # 先求p_t
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)  # 得到label的one_hot编码

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()  # 如果是多GPU训练 这里的cuda要指定搬运到指定GPU上 分布式多进程训练除外
        alpha = self.alpha[ids.data.view(-1)]
        # y*p_t  如果这里不用*， 还可以用gather提取出正确分到的类别概率。
        # 之所以能用sum，是因为class_mask已经把预测错误的概率清零了。
        probs = (P * class_mask).sum(1).view(-1, 1)
        # y*log(p_t)
        log_p = probs.log()
        # -a * (1-p_t)^2 * log(p_t)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
class BCELoss(nn.Module):
    def __init__(self, pos_weight=1, reduction='mean'):
        super(BCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, *], target: [N, *]
        # logits = F.sigmoid(logits)
        loss = - self.pos_weight * target * torch.log(logits+1e-5) - \
                (1 - target) * torch.log(1 - logits+1e-5)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
"8-neighbour loss"
class customedLoss(nn.Module):
    def __init__(self):
        super(customedLoss, self).__init__()
    def forward(self,out_image,segment_image,batch_num):
        loss = self.promising_node(out_image,segment_image,batch_num)

        return loss


    def promising_node(self,map_binary,segment_image,batch_num):
        loss = torch.tensor(0)
        loss = loss.float()
        map_binary=torch.squeeze(map_binary)
        segment_image=torch.squeeze(segment_image)
        for p in range(batch_num):
            map_binary_temp = map_binary[p]
            segment_image_temp = segment_image[p]
            truth_count = 0
            for i in range(map_binary_temp.shape[0]):
                for j in range(map_binary_temp.shape[1]):
                    # print(map_binary)
                    # print('*'*30)
                    # print(map_binary[0])
                    if map_binary_temp[i][j] >=0.8:
                        # print('*')
                        truth_count += 1
                        eight_adjacency_list_pred = self.eight_adjacency(map_binary_temp, i, j)
                        eight_adjacency_list_ture = self.eight_adjacency(segment_image_temp, i, j)
                        eight_weight = self.weight_update(segment_image_temp,i,j,eight_adjacency_list_ture)
                        # print(eight_weight)
                        # print(eight_adjacency_list_pred)
                        # print(eight_adjacency_list_ture)
                        temp1 = torch.tensor(0)
                        temp2 = torch.tensor(0)
                        for a in range(len(eight_weight)):
                            temp1 = temp1 + eight_weight[a]*torch.abs(eight_adjacency_list_ture[a]-eight_adjacency_list_pred[a])
                            # print("分子：%d"%temp1)
                            temp2 = temp2 + eight_weight[a]
                            # print("分母：%d" % temp2)
                        # print(type(temp2))
                        # print(type(temp1))
                        loss += (temp1)/(temp2+1)
                        # print(loss)
            loss = loss/truth_count
            # loss += torch.sum(torch.cat(torch.unsqueeze(torch.tensor([eight_weight[k]*(eight_adjacency_list_pred[k]-eight_adjacency_list_ture[k])] for k in range(len(eight_weight))),1)))/torch.sum(eight_weight)
        loss = loss/batch_num
        # print(loss)
        return loss

    def eight_adjacency(self,map_binary, i, j):
        x_max=map_binary.shape[0]
        y_max=map_binary.shape[1]

        eight_adjacency_list = torch.zeros(8)

        for a in range(2):
            if i>=1 and i<(x_max-1) and j>=1 and j<(y_max-1):
                eight_adjacency_list[0] = map_binary[i - 1][j - 1]
                eight_adjacency_list[1] = map_binary[i - 1][j]
                eight_adjacency_list[2] = map_binary[i - 1][j + 1]
                eight_adjacency_list[3] = map_binary[i][j - 1]
                eight_adjacency_list[4] = map_binary[i][j + 1]
                eight_adjacency_list[5] = map_binary[i + 1][j - 1]
                eight_adjacency_list[6] = map_binary[i + 1][j]
                eight_adjacency_list[7] = map_binary[i + 1][j + 1]
                break
            else:
                if i==0:
                    if j==0:
                        eight_adjacency_list[0] = 0
                        eight_adjacency_list[1] = 0
                        eight_adjacency_list[2] = 0
                        eight_adjacency_list[3] = 0
                        eight_adjacency_list[4] = map_binary[i][j + 1]
                        eight_adjacency_list[5] = 0
                        eight_adjacency_list[6] = map_binary[i + 1][j]
                        eight_adjacency_list[7] = map_binary[i + 1][j + 1]
                        break
                    if j==y_max-1:
                        eight_adjacency_list[0] = 0
                        eight_adjacency_list[1] = 0
                        eight_adjacency_list[2] = 0
                        eight_adjacency_list[3] = map_binary[i][j - 1]
                        eight_adjacency_list[4] = 0
                        eight_adjacency_list[5] = map_binary[i + 1][j - 1]
                        eight_adjacency_list[6] = map_binary[i + 1][j]
                        eight_adjacency_list[7] = 0
                        break
                    if j<y_max-1:
                        eight_adjacency_list[0] = 0
                        eight_adjacency_list[1] = 0
                        eight_adjacency_list[2] = 0
                        eight_adjacency_list[3] = map_binary[i][j - 1]
                        eight_adjacency_list[4] = map_binary[i][j + 1]
                        eight_adjacency_list[5] = map_binary[i + 1][j - 1]
                        eight_adjacency_list[6] = map_binary[i + 1][j]
                        eight_adjacency_list[7] = map_binary[i + 1][j + 1]
                        break
                if j==0:
                    if i==0:
                        eight_adjacency_list[0] = 0
                        eight_adjacency_list[1] = 0
                        eight_adjacency_list[2] = 0
                        eight_adjacency_list[3] = 0
                        eight_adjacency_list[4] = map_binary[i][j + 1]
                        eight_adjacency_list[5] = 0
                        eight_adjacency_list[6] = map_binary[i + 1][j]
                        eight_adjacency_list[7] = map_binary[i + 1][j + 1]
                        break
                    if i==x_max-1:
                        eight_adjacency_list[0] = 0
                        eight_adjacency_list[1] = map_binary[i - 1][j]
                        eight_adjacency_list[2] = map_binary[i - 1][j + 1]
                        eight_adjacency_list[3] = 0
                        eight_adjacency_list[4] = map_binary[i][j + 1]
                        eight_adjacency_list[5] = 0
                        eight_adjacency_list[6] = 0
                        eight_adjacency_list[7] = 0
                        break
                    if i<x_max-1:
                        eight_adjacency_list[0] = 0
                        eight_adjacency_list[1] = map_binary[i - 1][j]
                        eight_adjacency_list[2] = map_binary[i - 1][j + 1]
                        eight_adjacency_list[3] = 0
                        eight_adjacency_list[4] = map_binary[i][j + 1]
                        eight_adjacency_list[5] = 0
                        eight_adjacency_list[6] = map_binary[i + 1][j]
                        eight_adjacency_list[7] = map_binary[i + 1][j + 1]
                        break
        return eight_adjacency_list

    def weight_update(self,segment_image,i,j,eight_adjacency_list_ture):
        for t in range(8):
            eight_weight = torch.zeros(8)
            # eight_adjacency_list_ture = list(enumerate(eight_adjacency_list_ture))
            # position_list = torch.tensor([m for m,n in eight_adjacency_list_ture if n == 1])
            position_list = torch.nonzero(eight_adjacency_list_ture)
            for k in range(len(position_list)):
                if position_list[k] == 0:
                    i_new = i-1
                    j_new = j-1
                    partical_eight_adjacency_list_ture = self.eight_adjacency(segment_image, i_new, j_new)
                    eight_weight[0] = torch.sum(partical_eight_adjacency_list_ture)
                    continue
                if position_list[k] == 1:
                    i_new = i - 1
                    j_new = j
                    partical_eight_adjacency_list_ture = self.eight_adjacency(segment_image, i_new, j_new)
                    eight_weight[1] = torch.sum(partical_eight_adjacency_list_ture)
                    continue
                if position_list[k] == 2:
                    i_new = i - 1
                    j_new = j + 1
                    partical_eight_adjacency_list_ture = self.eight_adjacency(segment_image, i_new, j_new)
                    eight_weight[2] = torch.sum(partical_eight_adjacency_list_ture)
                    continue
                if position_list[k] == 3:
                    i_new = i
                    j_new = j - 1
                    partical_eight_adjacency_list_ture = self.eight_adjacency(segment_image, i_new, j_new)
                    eight_weight[3] = torch.sum(partical_eight_adjacency_list_ture)
                    continue
                if position_list[k] == 4:
                    i_new = i
                    j_new = j + 1
                    partical_eight_adjacency_list_ture = self.eight_adjacency(segment_image, i_new, j_new)
                    eight_weight[4] = torch.sum(partical_eight_adjacency_list_ture)
                    continue
                if position_list[k] == 5:
                    i_new = i + 1
                    j_new = j - 1
                    partical_eight_adjacency_list_ture = self.eight_adjacency(segment_image, i_new, j_new)
                    eight_weight[5] = torch.sum(partical_eight_adjacency_list_ture)
                    continue
                if position_list[k] == 6:
                    i_new = i + 1
                    j_new = j
                    partical_eight_adjacency_list_ture = self.eight_adjacency(segment_image, i_new, j_new)
                    eight_weight[6] = torch.sum(partical_eight_adjacency_list_ture)
                    continue
                if position_list[k] == 7:
                    i_new = i + 1
                    j_new = j + 1
                    partical_eight_adjacency_list_ture = self.eight_adjacency(segment_image, i_new, j_new)
                    eight_weight[7] = torch.sum(partical_eight_adjacency_list_ture)
                    continue
        return eight_weight

"soft dice loss"
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

"SignificanceLoss loss"
class SignificanceLoss(nn.Module):

    def __init__(self):
        super(SignificanceLoss, self).__init__()

    def forward(self, out_image, segment_image):
        loss = self.significance_matrix(out_image, segment_image)

        return loss
    def significance_matrix(self,out_image, segment_image):
        out_image = out_image.cuda()
        segment_image = segment_image.cuda()
        # out_binary = torch.ones_like(out_image).cuda()
        # out_binary[out_image < 0.5] = 0
        out_binary=out_image
        # [N,C,H,W]
        padding = (1, 1,
                   1, 1)
        out_binary_pad = functional.pad(out_binary, padding, mode='constant', value=0)
        segment_image_pad = functional.pad(segment_image, padding, mode='constant', value=0)

        out_binary_pad_reverse = torch.ones_like(out_binary_pad) - out_binary_pad
        segment_image_pad_reverse = torch.ones_like(segment_image_pad) - segment_image_pad

        kernel = torch.Tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).cuda()
        num = out_image.shape[0]
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        while kernel.shape[0] < num:
            kernel = torch.cat([kernel, kernel], dim=0)
        conv = nn.Conv2d(num, num, (3, 3), stride=1, bias=False).cuda()
        conv.weight.data = torch.Tensor(kernel)
        # conv.bias.data = torch.Tensor([1])

        sig_out_1 = 9 * torch.ones_like(out_image) - conv(out_binary_pad)
        sig_seg_1 = 9 * torch.ones_like(segment_image) - conv(segment_image_pad)

        sig_out_0 = 9 * torch.ones_like(out_image) - conv(out_binary_pad_reverse)
        sig_seg_0 = 9 * torch.ones_like(segment_image) - conv(segment_image_pad_reverse)

        M_temp_1 = torch.squeeze(sig_out_1)
        W_temp_1 = torch.squeeze(sig_seg_1)
        M_temp_0 = torch.squeeze(sig_out_0)
        W_temp_0 = torch.squeeze(sig_seg_0)
        loss = torch.abs(W_temp_1 - M_temp_1)+torch.abs(W_temp_0 - M_temp_0)
        loss = loss.mean()
        return loss

"weight BCE loss"
class WeightBCELoss1(nn.Module):
    def __init__(self):
        super(WeightBCELoss1, self).__init__()

    def forward(self, out_image, segment_image):
        loss = self.significance_matrix(out_image, segment_image)

        return loss
    def significance_matrix(self,out_image, segment_image):
        # out_binary = torch.ones_like(out_image).cuda()
        # out_binary[out_image < 0.5] = 0
        out_binary=out_image
        #[N,C,H,W]
        padding = (1, 1,
                   1, 1)
        out_binary_pad = functional.pad(out_binary, padding, mode='constant', value=0)
        segment_image_pad = functional.pad(segment_image, padding, mode='constant', value=0)

        out_binary_pad_reverse = torch.ones_like(out_binary_pad) - out_binary_pad
        segment_image_pad_reverse = torch.ones_like(segment_image_pad) - segment_image_pad


        kernel = torch.Tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        num = out_image.shape[0]
        kernel=kernel.unsqueeze(0)
        kernel=kernel.unsqueeze(0)
        while kernel.shape[0]<num:
            kernel=torch.cat([kernel,kernel],dim=0)
        kernel=kernel.cuda()
        conv = nn.Conv2d(num, num, (3, 3), stride = 1, bias=False)
        conv.weight.data = torch.Tensor(kernel)
        # conv.bias.data = torch.Tensor([1])
        # print(out_binary_pad.shape)
        # print(conv(out_binary_pad).shape)
        # print(torch.ones_like(out_image).shape)

        sig_out_1 = 9 * torch.ones_like(out_image)-conv(out_binary_pad)
        sig_seg_1 = 9 * torch.ones_like(segment_image)-conv(segment_image_pad)

        sig_out_0 = 9 * torch.ones_like(out_image) - conv(out_binary_pad_reverse)
        sig_seg_0 = 9 * torch.ones_like(segment_image) - conv(segment_image_pad_reverse)


        M_temp_1 = torch.squeeze(sig_out_1)
        W_temp_1 = torch.squeeze(sig_seg_1)
        M_temp_0 = torch.squeeze(sig_out_0)
        W_temp_0 = torch.squeeze(sig_seg_0)
        seg_temp = torch.squeeze(segment_image)
        out_temp = torch.squeeze(out_image)

        loss = -((torch.abs(W_temp_1-M_temp_1)+1) * seg_temp * torch.log(out_temp+1e-5) + (torch.abs(W_temp_0-M_temp_0)+1) * (
                            1 - seg_temp) * torch.log(1 - out_temp+1e-5))

        loss = loss.mean()

        return loss

"weight BCE loss 2"
class WeightBCELoss2(nn.Module):

    def __init__(self):
        super(WeightBCELoss2, self).__init__()

    def forward(self, out_image, segment_image):
        loss = self.significance_matrix(out_image, segment_image)

        return loss
    def significance_matrix(self,out_image, segment_image):
        # out_binary = torch.ones_like(out_image).cuda()
        # out_binary[out_image < 0.5] = 0
        out_binary=out_image
        #[N,C,H,W]
        padding = (1, 1,
                   1, 1)
        out_binary_pad = functional.pad(out_binary, padding, mode='constant', value=0)
        segment_image_pad = functional.pad(segment_image, padding, mode='constant', value=0)

        out_binary_pad_reverse = torch.ones_like(out_binary_pad) - out_binary_pad


        kernel = torch.Tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        num = out_image.shape[0]
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        while kernel.shape[0] < num:
            kernel = torch.cat([kernel, kernel], dim=0)
        kernel = kernel.cuda()
        conv = nn.Conv2d(num, num, (3, 3), stride=1, bias=False)
        conv.weight.data = torch.Tensor(kernel)
        # conv.bias.data = torch.Tensor([1])

        sig_seg_1 = 9 * torch.ones_like(segment_image)-conv(segment_image_pad)

        sig_out_0 = 9 * torch.ones_like(out_image) - conv(out_binary_pad_reverse)

        W_temp_1 = torch.squeeze(sig_seg_1)
        M_temp_0 = torch.squeeze(sig_out_0)
        seg_temp = torch.squeeze(segment_image)
        out_temp = torch.squeeze(out_image)

        loss = -(W_temp_1 * seg_temp * torch.log(out_temp+1e-5) + M_temp_0 * (
                            1 - seg_temp) * torch.log(1 - out_temp+1e-5))

        loss = loss.mean()
        return loss

"weight BCE loss 3"
class WeightBCELoss3(nn.Module):

    def __init__(self):
        super(WeightBCELoss3, self).__init__()

    def forward(self, out_image, segment_image):
        loss = self.significance_matrix(out_image, segment_image)

        return loss
    def significance_matrix(self,out_image, segment_image):
        # out_binary = torch.ones_like(out_image).cuda()
        # out_binary[out_image < 0.5] = 0
        out_binary=out_image
        #[N,C,H,W]
        padding = (1, 1,
                   1, 1)
        out_binary_pad = functional.pad(out_binary, padding, mode='constant', value=0)
        segment_image_pad = functional.pad(segment_image, padding, mode='constant', value=0)

        kernel = torch.Tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        num = out_image.shape[0]
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        while kernel.shape[0] < num:
            kernel = torch.cat([kernel, kernel], dim=0)
        kernel = kernel.cuda()
        conv = nn.Conv2d(num, num, (3, 3), stride=1, bias=False)
        conv.weight.data = torch.Tensor(kernel)
        # conv.bias.data = torch.Tensor([1])
        sig_seg_1 = 9 * torch.ones_like(segment_image)-conv(segment_image_pad)
        W_temp_1 = torch.squeeze(sig_seg_1)

        seg = torch.squeeze(segment_image)
        out = torch.squeeze(out_image)

        out_temp = out
        seg_temp = seg

        loss = -(W_temp_1 * seg_temp * torch.log(out_temp+1e-5) + (
                            1 - seg_temp) * torch.log(1 - out_temp+1e-5))


        loss = loss.mean()
        return loss

"weight BCE loss 4"
class WeightBCELoss4(nn.Module):

    def __init__(self):
        super(WeightBCELoss4, self).__init__()

    def forward(self, out_image, segment_image):
        loss = self.significance_matrix(out_image, segment_image)

        return loss
    def significance_matrix(self,out_image, segment_image):
        # out_binary = torch.ones_like(out_image).cuda()
        # out_binary[out_image < 0.5] = 0
        out_binary = out_image
        #[N,C,H,W]
        padding = (1, 1,
                   1, 1)
        out_binary_pad = functional.pad(out_binary, padding, mode='constant', value=0)
        segment_image_pad = functional.pad(segment_image, padding, mode='constant', value=0)

        out_binary_pad_reverse = torch.ones_like(out_binary_pad) - out_binary_pad
        segment_image_pad_reverse = torch.ones_like(segment_image_pad) - segment_image_pad


        kernel = torch.Tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        num = out_image.shape[0]
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        while kernel.shape[0] < num:
            kernel = torch.cat([kernel, kernel], dim=0)
        kernel = kernel.cuda()
        conv = nn.Conv2d(num, num, (3, 3), stride=1, bias=False)
        conv.weight.data = torch.Tensor(kernel)
        # conv.bias.data = torch.Tensor([1])

        sig_seg_1 = conv(segment_image_pad)
        W_temp_1 = torch.squeeze(sig_seg_1)

        seg = torch.squeeze(segment_image)
        out = torch.squeeze(out_image)

        out_temp = out
        seg_temp = seg

        loss = -(W_temp_1 * seg_temp * torch.log(out_temp+1e-5) + W_temp_1 *(
                            1 - seg_temp) * torch.log(1 - out_temp+1e-5))


        loss = loss.mean()
        return loss

"weight/hard or easy BCE loss"
class Weight_EH_BCELoss(nn.Module):

    def __init__(self):
        super(Weight_EH_BCELoss, self).__init__()

    def forward(self, out_image, segment_image):
        loss = self.significance_matrix_EH(out_image, segment_image)

        return loss
    def significance_matrix_EH(self,out_image, segment_image):
        # out_binary = torch.ones_like(out_image).cuda()
        # out_binary[out_image < 0.5] = 0
        out_binary=out_image
        # [N,C,H,W]
        padding = (1, 1,
                   1, 1)
        out_binary_pad = functional.pad(out_binary, padding, mode='constant', value=0)
        segment_image_pad = functional.pad(segment_image, padding, mode='constant', value=0)

        out_binary_pad_reverse = torch.ones_like(out_binary_pad) - out_binary_pad
        segment_image_pad_reverse = torch.ones_like(segment_image_pad) - segment_image_pad

        kernel = torch.Tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        num = out_image.shape[0]
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        while kernel.shape[0] < num:
            kernel = torch.cat([kernel, kernel], dim=0)
        kernel = kernel.cuda()
        conv = nn.Conv2d(num, num, (3, 3), stride=1, bias=False)
        conv.weight.data = torch.Tensor(kernel)
        # conv.bias.data = torch.Tensor([1])

        sig_out_1 = 9 * torch.ones_like(out_image) - conv(out_binary_pad)
        sig_seg_1 = 9 * torch.ones_like(segment_image) - conv(segment_image_pad)

        sig_out_0 = 9 * torch.ones_like(out_image) - conv(out_binary_pad_reverse)
        sig_seg_0 = 9 * torch.ones_like(segment_image) - conv(segment_image_pad_reverse)

        M_temp_1 = torch.squeeze(sig_out_1)
        W_temp_1 = torch.squeeze(sig_seg_1)
        M_temp_0 = torch.squeeze(sig_out_0)
        W_temp_0 = torch.squeeze(sig_seg_0)
        seg = torch.squeeze(segment_image)
        out = torch.squeeze(out_image)

        out_temp = out
        seg_temp = seg

        loss = -((torch.abs(W_temp_1 - M_temp_1) + 1) * torch.square_(1 - out_temp) * seg_temp * torch.log(
                        out_temp+1e-5) + (torch.abs(W_temp_0 - M_temp_0) + 1) * torch.square_(out_temp) * (
                                      1 - seg_temp) * torch.log(1 - out_temp+1e-5))

        loss = loss.mean()
        return loss

# reverse: emphasis on. the center of the target instead of the edge


"SignificanceLoss loss reverse"
class SignificanceLoss_reverse(nn.Module):

    def __init__(self):
        super(SignificanceLoss_reverse, self).__init__()

    def forward(self, out_image, segment_image):
        loss = self.significance_matrix(out_image, segment_image)

        return loss
    def significance_matrix(self,out_image, segment_image):

        # '''hard'''
        # out_binary = torch.ones_like(out_image)
        # out_binary[out_image < 0.5] = 0
        "soft"
        out_binary=out_image
        # [N,C,H,W]
        padding = (1, 1,
                   1, 1)
        out_binary_pad = functional.pad(out_binary, padding, mode='constant', value=0)
        segment_image_pad = functional.pad(segment_image, padding, mode='constant', value=0)

        # out_binary_pad_reverse = torch.ones_like(out_binary_pad) - out_binary_pad
        # segment_image_pad_reverse = torch.ones_like(segment_image_pad) - segment_image_pad

        kernel = torch.Tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        num = out_image.shape[0]
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        while kernel.shape[0] < num:
            kernel = torch.cat([kernel, kernel], dim=0)
        kernel = kernel.cuda()
        conv = nn.Conv2d(num, num, (3, 3), stride=1, bias=False)
        conv.weight.data = torch.Tensor(kernel)
        # conv.bias.data = torch.Tensor([1])

        sig_out_1 = conv(out_binary_pad)
        sig_seg_1 = conv(segment_image_pad)

        # sig_out_0 = conv(out_binary_pad_reverse)
        # sig_seg_0 = conv(segment_image_pad_reverse)

        M_temp_1 = torch.squeeze(sig_out_1)
        W_temp_1 = torch.squeeze(sig_seg_1)
        # M_temp_0 = torch.squeeze(sig_out_0)
        # W_temp_0 = torch.squeeze(sig_seg_0)

        #original
        loss = 2*torch.abs(W_temp_1 - M_temp_1)
        #only 1
        # loss = torch.abs(W_temp_1 - M_temp_1)
        # alpha1
        # alpha = 0.5
        # loss = alpha*(torch.abs(W_temp_1 - M_temp_1)+torch.abs(W_temp_0 - M_temp_0))
        # alpha2
        # loss = alpha*torch.abs(W_temp_1 - M_temp_1)

        loss = loss.mean()
        return loss
"Purity BCE loss"
class PurityBCE(nn.Module):

    def __init__(self):
        super(PurityBCE, self).__init__()

    def forward(self, out_image, segment_image):
        loss = self.significance_matrix(out_image, segment_image)

        return loss
    def significance_matrix(self,out_image, segment_image):
        '''hard'''
        # out_binary = torch.ones_like(out_image)
        # out_binary[out_image < 0.5] =
        '''soft'''
        out_binary = out_image
        #[N,C,H,W]
        padding = (1, 1,
                   1, 1)
        out_binary_pad = functional.pad(out_binary, padding, mode='constant', value=0)
        segment_image_pad = functional.pad(segment_image, padding, mode='constant', value=0)

        out_binary_pad_reverse = torch.ones_like(out_binary_pad) - out_binary_pad
        segment_image_pad_reverse = torch.ones_like(segment_image_pad) - segment_image_pad

        kernel = torch.Tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        num = out_image.shape[0]
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        while kernel.shape[0] < num:
            kernel = torch.cat([kernel, kernel], dim=0)
        kernel = kernel.cuda()
        conv = nn.Conv2d(num, num, (3, 3), stride=1, bias=False)
        conv.weight.data = torch.Tensor(kernel)
        # conv.bias.data = torch.Tensor([1])

        sig_out_1 = conv(out_binary_pad)
        sig_seg_1 = conv(segment_image_pad)

        sig_out_0 = conv(out_binary_pad_reverse)
        sig_seg_0 = conv(segment_image_pad_reverse)

        M_temp_1 = torch.squeeze(sig_out_1)
        W_temp_1 = torch.squeeze(sig_seg_1)
        M_temp_0 = torch.squeeze(sig_out_0)
        W_temp_0 = torch.squeeze(sig_seg_0)

        # loss = -((torch.abs(W_temp_1-M_temp_1)+1) * seg_temp * torch.log(out_temp+1e-5) + (torch.abs(W_temp_0-M_temp_0)+1) * (
        #                     1 - seg_temp) * torch.log(1 - out_temp+1e-5))
        # loss = -((W_temp_1 + 1) * seg_temp * torch.log(out_temp + 1e-5) + (
        #                  1 - seg_temp) * torch.log(1 - out_temp + 1e-5))
        loss = -W_temp_1/8*torch.log(M_temp_1/8+1e-5)-W_temp_0/8*torch.log(M_temp_0/8+1e-5)
        loss = loss.mean()
        return loss
"weight BCE loss reverse"
class WeightBCELoss1_reverse(nn.Module):

    def __init__(self):
        super(WeightBCELoss1_reverse, self).__init__()

    def forward(self, out_image, segment_image):
        loss = self.significance_matrix(out_image, segment_image)

        return loss
    def significance_matrix(self,out_image, segment_image):
        '''hard'''
        # out_binary = torch.ones_like(out_image)
        # out_binary[out_image < 0.5] =
        '''soft'''
        out_binary = out_image
        #[N,C,H,W]
        padding = (1, 1,
                   1, 1)
        out_binary_pad = functional.pad(out_binary, padding, mode='constant', value=0)
        segment_image_pad = functional.pad(segment_image, padding, mode='constant', value=0)

        # out_binary_pad_reverse = torch.ones_like(out_binary_pad) - out_binary_pad
        # segment_image_pad_reverse = torch.ones_like(segment_image_pad) - segment_image_pad


        kernel = torch.Tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        num = out_image.shape[0]
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)

        while kernel.shape[0] < num:
            kernel = torch.cat([kernel, kernel], dim=0)
        kernel = kernel.cuda()
        conv = nn.Conv2d(num, num, (3, 3), stride=1, bias=False)
        conv.weight.data = torch.Tensor(kernel)
        # conv.bias.data = torch.Tensor([1])

        sig_out_1 = conv(out_binary_pad)
        sig_seg_1 = conv(segment_image_pad)

        # sig_out_0 = conv(out_binary_pad_reverse)
        # sig_seg_0 = conv(segment_image_pad_reverse)


        M_temp_1 = torch.squeeze(sig_out_1)
        W_temp_1 = torch.squeeze(sig_seg_1)
        # M_temp_0 = torch.squeeze(sig_out_0)
        # W_temp_0 = torch.squeeze(sig_seg_0)
        seg = torch.squeeze(segment_image)
        out = torch.squeeze(out_image)

        out_temp = out
        seg_temp = seg

        loss = -((torch.abs(W_temp_1-M_temp_1)+1) * seg_temp * torch.log(out_temp+1e-5) + (torch.abs(W_temp_1-M_temp_1)+1) * (
                            1 - seg_temp) * torch.log(1 - out_temp+1e-5))
        # loss = -((W_temp_1 + 1) * seg_temp * torch.log(out_temp + 1e-5) + (
        #                  1 - seg_temp) * torch.log(1 - out_temp + 1e-5))
        loss = loss.mean()
        return loss

"weight BCE loss 2 reverse"
class WeightBCELoss2_reverse(nn.Module):

    def __init__(self):
        super(WeightBCELoss2_reverse, self).__init__()

    def forward(self, out_image, segment_image):
        loss = self.significance_matrix(out_image, segment_image)

        return loss
    def significance_matrix(self,out_image, segment_image):
        # out_binary = torch.ones_like(out_image).cuda()
        # out_binary[out_image < 0.5] = 0
        out_binary = out_image
        #[N,C,H,W]
        padding = (1, 1,
                   1, 1)
        out_binary_pad = functional.pad(out_binary, padding, mode='constant', value=0)
        segment_image_pad = functional.pad(segment_image, padding, mode='constant', value=0)

        out_binary_pad_reverse = torch.ones_like(out_binary_pad) - out_binary_pad
        segment_image_pad_reverse = torch.ones_like(segment_image_pad) - segment_image_pad


        kernel = torch.Tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        num = out_image.shape[0]
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        while kernel.shape[0] < num:
            kernel = torch.cat([kernel, kernel], dim=0)
        kernel = kernel.cuda()
        conv = nn.Conv2d(num, num, (3, 3), stride=1, bias=False)
        conv.weight.data = torch.Tensor(kernel)
        # conv.bias.data = torch.Tensor([1])

        sig_out_1 = conv(out_binary_pad)
        sig_seg_1 = conv(segment_image_pad)

        sig_out_0 = conv(out_binary_pad_reverse)
        sig_seg_0 = conv(segment_image_pad_reverse)

        M_temp_1 = torch.squeeze(sig_out_1)
        W_temp_1 = torch.squeeze(sig_seg_1)
        M_temp_0 = torch.squeeze(sig_out_0)
        W_temp_0 = torch.squeeze(sig_seg_0)
        seg = torch.squeeze(segment_image)
        out = torch.squeeze(out_image)

        out_temp = out
        seg_temp = seg

        loss = -(W_temp_1 * seg_temp * torch.log(out_temp+1e-5) + M_temp_0 * (
                            1 - seg_temp) * torch.log(1 - out_temp+1e-5))


        loss = loss.mean()
        return loss

"weight BCE loss 3 reverse"
class WeightBCELoss3_reverse(nn.Module):

    def __init__(self):
        super(WeightBCELoss3_reverse, self).__init__()

    def forward(self, out_image, segment_image):
        loss = self.significance_matrix(out_image, segment_image)

        return loss
    def significance_matrix(self,out_image, segment_image):
        # out_binary = torch.ones_like(out_image).cuda()
        # out_binary[out_image < 0.5] = 0
        out_binary = out_image
        #[N,C,H,W]
        padding = (1, 1,
                   1, 1)
        out_binary_pad = functional.pad(out_binary, padding, mode='constant', value=0)
        segment_image_pad = functional.pad(segment_image, padding, mode='constant', value=0)

        out_binary_pad_reverse = torch.ones_like(out_binary_pad) - out_binary_pad
        segment_image_pad_reverse = torch.ones_like(segment_image_pad) - segment_image_pad


        kernel = torch.Tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        num = out_image.shape[0]
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        while kernel.shape[0] < num:
            kernel = torch.cat([kernel, kernel], dim=0)
        kernel = kernel.cuda()
        conv = nn.Conv2d(num, num, (3, 3), stride=1, bias=False)
        conv.weight.data = torch.Tensor(kernel)
        # conv.bias.data = torch.Tensor([1])

        sig_seg_1 = conv(segment_image_pad)
        W_temp_1 = torch.squeeze(sig_seg_1)

        seg = torch.squeeze(segment_image)
        out = torch.squeeze(out_image)

        out_temp = out
        seg_temp = seg

        loss = -(W_temp_1 * seg_temp * torch.log(out_temp+1e-5) + (
                            1 - seg_temp) * torch.log(1 - out_temp+1e-5))


        loss = loss.mean()
        return loss


"weight/hard or easy BCE loss reverse"
class Weight_EH_BCELoss_reverse(nn.Module):

    def __init__(self):
        super(Weight_EH_BCELoss_reverse, self).__init__()

    def forward(self, out_image, segment_image):
        loss = self.significance_matrix_EH(out_image, segment_image)

        return loss
    def significance_matrix_EH(self,out_image, segment_image):
        # out_binary = torch.ones_like(out_image).cuda()
        # out_binary[out_image < 0.5] = 0
        out_binary = out_image
        # [N,C,H,W]
        padding = (1, 1,
                   1, 1)
        out_binary_pad = functional.pad(out_binary, padding, mode='constant', value=0)
        segment_image_pad = functional.pad(segment_image, padding, mode='constant', value=0)

        out_binary_pad_reverse = torch.ones_like(out_binary_pad) - out_binary_pad
        segment_image_pad_reverse = torch.ones_like(segment_image_pad) - segment_image_pad

        kernel = torch.Tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        num = out_image.shape[0]
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        while kernel.shape[0] < num:
            kernel = torch.cat([kernel, kernel], dim=0)
        kernel = kernel.cuda()
        conv = nn.Conv2d(num, num, (3, 3), stride=1, bias=False)
        conv.weight.data = torch.Tensor(kernel)
        # conv.bias.data = torch.Tensor([1])

        sig_out_1 = conv(out_binary_pad)
        sig_seg_1 = conv(segment_image_pad)

        sig_out_0 = conv(out_binary_pad_reverse)
        sig_seg_0 = conv(segment_image_pad_reverse)

        M_temp_1 = torch.squeeze(sig_out_1)
        W_temp_1 = torch.squeeze(sig_seg_1)
        M_temp_0 = torch.squeeze(sig_out_0)
        W_temp_0 = torch.squeeze(sig_seg_0)
        seg = torch.squeeze(segment_image)
        out = torch.squeeze(out_image)

        out_temp = out
        seg_temp = seg
        # 1
        # loss = -((torch.abs(W_temp_1 - M_temp_1) + 1) * seg_temp * torch.log(
        #                 out_temp+1e-5) + (torch.abs(W_temp_0 - M_temp_0) + 1) *(out_temp) * (
        #                               1 - seg_temp) * torch.log(1 - out_temp+1e-5))
        # 3
        # loss = -(W_temp_1 * seg_temp * torch.log(
        #     out_temp + 1e-5) + W_temp_1 * out_temp * (1 - seg_temp) * torch.log(1 - out_temp + 1e-5))
        # 2
        # loss = -((torch.abs(W_temp_1 - M_temp_1) + 1) * seg_temp * torch.log(
        #     out_temp + 1e-5) + (out_temp) * (1 - seg_temp) * torch.log(1 - out_temp + 1e-5))
        # 4
        # loss = -(W_temp_1 * seg_temp * torch.log(
        #     out_temp + 1e-5) + (out_temp) * (1 - seg_temp) * torch.log(1 - out_temp + 1e-5))
        # 5
        loss = -(W_temp_1 * seg_temp * torch.log(
            out_temp + 1e-5) + (1 - seg_temp) * torch.log(1 - out_temp + 1e-5))

        loss = loss.mean()
        return loss



"SSIM loss"
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
def _logssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    ssim_map = (ssim_map - torch.min(ssim_map))/(torch.max(ssim_map)-torch.min(ssim_map))
    ssim_map = -torch.log(ssim_map + 1e-8)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
class LOGSSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(LOGSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _logssim(img1, img2, window, self.window_size, channel, self.size_average)
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)