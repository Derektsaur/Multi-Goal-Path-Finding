import csv
import random
import time

import torch.nn
from torch import nn,optim
from torch.utils.data import DataLoader
from PIL import Image
from data import *
# from net import *
# from Customed_Loss import *
from utils_2 import *
from utils import MTLLoss
from Custom_Loss import *

from Regression import *
from RegModels import *

#BCE 针对pixel进行平均
#DICE 针对map
#只训练seg out的bce 和 dice 是可行的

'''hyperparameters'''
batch_num = 16
batch_num_validate = 16
batch_num_test = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# weight_path= 'params/74.pth'
data_path='Data_TSP'
save_weight_path='params/unet_%d.pth'
save_path='train_image_mine'
if __name__ == '__main__':
    torch.cuda.empty_cache()
    custom_dataset = MyDataset(data_path)
    train_size = int(len(custom_dataset) * 0.6) #0.75:0.125:0.125
    validate_size = int(len(custom_dataset) * 0.2)
    test_size = int(len(custom_dataset))-train_size-validate_size
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(custom_dataset,[train_size, validate_size, test_size], generator=torch.Generator().manual_seed(0))# 随机将数据以一定比例分为train，validate，test


    data_loader = DataLoader(train_dataset, batch_size=batch_num, shuffle=True,  pin_memory=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_num_validate, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_num_test, shuffle=False, pin_memory=True)
    seed = 42
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # data_loader=DataLoa
    # data_loader=DataLoader(MyDataset(data_path),batch_size=batch_num,shuffle=True, num_workers=8, pin_memory=True)
    # input: tensor[3*H*W] in [0-1.0]  gt:tensor[1*H*W] in [0-1.0]
    # print(len(data_loader))
    # net=UNet_SE4_concat2().to(device)
    torch.cuda.empty_cache()
    net = SRegNet_mod3().to(device)
    best_epoch = int(input("Please input the pretrained epoch weight"))
    weights_pre = r'C:\Users\huang\Desktop\S&Reg V2\TRAIN MODEL\best result/best_%d.pth' % best_epoch
    if os.path.exists(weights_pre):
        net.load_state_dict(torch.load(weights_pre))
        opt = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)
        print('successfully load weights [pretrained]')
    else:
        print('unsuccessfully load weights [pretrained]')
    net.train()

    loss_fun1 = FocalLoss(alpha=0.25,gamma=2)
    loss_fun2 = SoftDiceLoss()
    loss_fun3 = nn.MSELoss(reduction="mean")
    loss_fun4 = nn.BCELoss()
    loss_fun5 = FocalLoss(alpha=2,gamma=0)


    # opt=optim.Adam(net.parameters(),lr=0.0001)
    # schedular = CosineAnnealingWarmRestarts(opt,T_0=1,T_mult=2)


    epoch  = 1
    epoch_num = 65
    # count = 0
    # early_stop_count = 5
    #    evaluation_criteria = [recall, precision, acc, fnr, F1, IOU, DICE]
    ave_Loss = []
    ave_recall = []
    ave_precision = []
    ave_acc = []
    ave_fnr = []
    ave_F1 = []
    ave_IOU = []
    ave_DICE = []
    ave_Loss_validate = []
    ave_recall_validate = []
    ave_precision_validate = []
    ave_acc_validate = []
    ave_fnr_validate = []
    ave_F1_validate = []
    ave_IOU_validate = []
    ave_DICE_validate = []
    ave_evaluation_criteria_validate_pre =[-1,-1,-1,-1,-1,-1,-1]
    ave_evaluation_criteria_validate = [0, 0, 0, 0, 0, 0, 0]

    ave_loss1 = []
    ave_loss2 = []
    ave_loss3 = []
    alpha = 2
    # while epoch < epoch_num:
    #
    #     since = time.time()
    #     torch.cuda.empty_cache()
    #
    #     net.train()
    #     total_Loss = 0
    #     total_batch_Loss = 0
    #     total_loss1 = 0
    #     total_loss2 = 0
    #     total_loss3 = 0
    #     total_evaluation_criteria = [0, 0, 0, 0, 0, 0, 0]
    #     total_batch_evaluation_criteria = [0, 0, 0, 0, 0, 0, 0]
    #     for i,(image, segment_image1, segment_image2, cost, coord) in enumerate(data_loader):
    #         # i : 0~len(data_loader.dataset)/batch_num=len(data_loader)
    #         # bs=16, 总16000，每次输入16张，共750次循环，每75次输出batch ave loss
    #         image, segment_image1, segment_image2, cost = image.to(device), segment_image1.to(device), segment_image2.to(device), cost.to(device)
    #         out_image1, out_image2, out_cost = net(image)
    #
    #
    #         # for param_tensor in net.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
    #         #     print(param_tensor, '\t', net.state_dict()[param_tensor].size())
    #         # W1 = net.state_dict()['fc1.weight']
    #         # print(out_cost.view(-1),cost.view(-1))
    #
    #         train_loss1 = loss_fun3(torch.squeeze(out_cost,dim=1),cost).to(device)
    #
    #         train_loss = train_loss1
    #
    #         opt.zero_grad()
    #         train_loss.backward()
    #         opt.step()
    #         # Criteria for evaluation
    #
    #
    #
    #         total_Loss += train_loss.item()
    #         total_batch_Loss += train_loss.item()
    #
    #
    #         '''analysis'''
    #         #data_loader len: samples number in each batch
    #         #data_loader.dataser len: samplers number totally
    #         if i == 0:
    #             print('start epoch: %d'%epoch)
    #         if i % (len(data_loader) / 10) == (len(data_loader) / 10) - 1:
    #             if i != 0:
    #                 print(
    #                     "epoch: {} [{}/{} ({:0f}%)]---loss: {:.4f}".format(
    #                         epoch, (i + 1) * len(image), len(data_loader.dataset),
    #                         100. * (i + 1) / len(data_loader), total_batch_Loss / (len(data_loader) / 10)))
    #
    #                 total_batch_Loss = 0
    #             # print('Epoch ' + str(epoch) + ' : ' + str(i // 200) + ' , LOSS =' + str(train_loss.item()))
    #         if i==len(data_loader)-1:
    #             print("epoch: {}--ave loss: {:.6f}".format(epoch, total_Loss/(len(data_loader))))
    #             #    evaluation_criteria = [recall, precision, acc, fnr, F1, IOU, DICE]
    #             ave_Loss.append(total_Loss/(len(data_loader)))
    #
    #
    #         '''save weights'''
    #         if i==len(data_loader)-1:
    #             torch.save(net.state_dict(),save_weight_path%epoch)
    #
    #     '''test on validate dataset'''
    #     print('******test on validate dataset******')
    #     torch.cuda.empty_cache()
    #     net.eval()
    #     total_Loss_validate = 0
    #     total_evaluation_criteria_validate = [0, 0, 0, 0, 0, 0, 0]
    #     for i, (image, segment_image1, segment_image2, cost, coord) in enumerate(validate_loader):
    #         # i : 0~len(data_loader.dataset)/batch_num=len(data_loader)
    #         # bs=16, 总16000，每次输入16张，共750次循环，每75次输出batch ave loss
    #         image, segment_image1, segment_image2, cost = image.to(device), segment_image1.to(
    #             device), segment_image2.to(device), cost.to(device)
    #         with torch.no_grad():
    #             out_image1, out_image2, out_cost = net(image)
    #             # for param_tensor in net.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
    #             #     print(param_tensor, '\t', net.state_dict()[param_tensor].size())
    #             # W1 = net.state_dict()['fc1.weight']
    #             # print(out_cost.view(-1),cost.view(-1))
    #             train_loss1 = loss_fun3(torch.squeeze(out_cost, dim=1), cost).to(device)
    #
    #             train_loss = train_loss1
    #
    #         # Criteria for evaluation
    #         total_Loss_validate += train_loss.item()
    #
    #
    #         #    evaluation_criteria = [recall, precision, acc, fnr, F1, IOU, DICE]
    #
    #     loss = total_Loss_validate / (len(validate_loader))
    #
    #
    #     print(
    #         "[test result on validate dataset]--ave loss: {:.6f}".format(loss))
    #     ave_Loss_validate.append(loss)
    #
    # #    evaluation_criteria = [recall, precision, acc, fnr, F1, IOU, DICE]
    #     time_elapsed = time.time() - since
    #     print('Epoch {} training complete in {:.0f}m {:.0f}s'.format(epoch,
    #                                                                  time_elapsed // 60, time_elapsed % 60))
    #     epoch += 1 #:D
    # #
    # # # break_epoch = epoch - 1 #:D
    # break_epoch = ave_Loss_validate.index(min(ave_Loss_validate))+1
    # # break_epoch=58
    #
    # epoch = epoch - 1 #:D
    #
    # draw_comparision(ave_Loss,ave_Loss_validate,epoch,label1='training dataset',label2='validate dataset',index_x='epoch',index_y='loss',title='loss-epoch3')

    '''test 1 on test dataset'''
    print('******test 1 on test dataset******')
    net.eval()
    with open(F'TestResult/cost_result.csv','w',newline='') as csvfile:
        f = csv.writer(csvfile)
        f.writerow(["index","prediction","ground truth"])
    torch.cuda.empty_cache()
    # test_1_data_path = r'C:\Users\huang\Desktop\Region A star\data\test1'
    # randomed_data_loader=DataLoader(MyDataset(test_1_data_path),batch_size=batch_num_test,shuffle=False)
    total_Loss_test = 0
    eva1 = [0,0,0,0,0,0,0]
    eva2 = [0,0,0,0,0,0,0]
    total_evaluation_criteria_test = [0, 0, 0, 0, 0, 0, 0]
    ave_evaluation_criteria_test = [0, 0, 0, 0, 0, 0, 0]
    weights_pre = r'C:\Users\huang\Desktop\S&Reg V2\TRAIN MODEL\best result/best_%d.pth' % best_epoch
    if os.path.exists(weights_pre):
        net.load_state_dict(torch.load(weights_pre))
        print('successfully load weights')
        print('the break epoch is {}'.format(weights_pre))
    else:
        print('unsuccessfully load weights')
    for l, (image, segment_image1, segment_image2, cost, coord) in enumerate(test_loader):
        # i : 0~len(data_loader.dataset)/batch_num=len(data_loader)
        # bs=16, 总16000，每次输入16张，共750次循环，每75次输出batch ave loss
        with torch.no_grad():

            image, segment_image1, segment_image2, cost = image.to(device), segment_image1.to(device), segment_image2.to(
                device), cost.to(device)
            out_image1, out_image2, out_cost = net(image)
            train_loss1 = loss_fun3(torch.squeeze(out_cost,dim=1),cost).to(device)

            train_loss = train_loss1


        for j in range(batch_num_test):
            img = torch.squeeze(image[j])
            predict_img = torch.squeeze(out_image2[j])
            predict_img1 = torch.stack([predict_img, predict_img], dim=0)
            true_img = torch.squeeze(segment_image2[j])
            true_img1 = torch.stack([true_img, true_img], dim=0)

            save_image_tensor2pillow(img, F'TestResult/Result1/map{l}_{j}.jpg')
            save_image_tensor2pillow(predict_img1, F'TestResult/Result1/2_prediction{l}_{j}.jpg')
            save_image_tensor2pillow(true_img1, F'TestResult/Result1/2_groundtruth{l}_{j}.jpg')

            original_img = image[j]
            unit_image1 = torch.stack([predict_img, true_img], dim=0)
            unit_image2 = torch.stack([original_img[0] - 110 / 255 * torch.squeeze(predict_img),
                                       original_img[1] - torch.squeeze(predict_img),
                                       original_img[2] - torch.squeeze(predict_img)], dim=0)
            unit_image2 = torch.squeeze(unit_image2)

            save_image_tensor2pillow(unit_image2, F'TestResult/Result1/2_{l}_{j}_original.jpg')
            save_image_tensor2pillow(unit_image1, F'TestResult/Result1/2_{l}_{j}.jpg')

        for j in range(batch_num_test):
            img = torch.squeeze(image[j])
            predict_img = torch.squeeze(out_image1[j])
            predict_img1 = torch.stack([predict_img, predict_img], dim=0)
            true_img = torch.squeeze(segment_image1[j])
            true_img1 = torch.stack([true_img, true_img], dim=0)

            save_image_tensor2pillow(predict_img1, F'TestResult/Result1/1_prediction{l}_{j}.jpg')
            save_image_tensor2pillow(true_img1, F'TestResult/Result1/1_groundtruth{l}_{j}.jpg')

            original_img = image[j]
            unit_image1 = torch.stack([predict_img, true_img], dim=0)
            unit_image2 = torch.stack([original_img[0] - 110 / 255 * torch.squeeze(predict_img),
                                       original_img[1] - torch.squeeze(predict_img),
                                       original_img[2] - torch.squeeze(predict_img)], dim=0)
            unit_image2 = torch.squeeze(unit_image2)

            save_image_tensor2pillow(unit_image2, F'TestResult/Result1/1_{l}_{j}_original.jpg')
            save_image_tensor2pillow(unit_image1, F'TestResult/Result1/1_{l}_{j}.jpg')
            with open(F'TestResult/cost_result.csv', 'a',newline='') as csvfile:
                f = csv.writer(csvfile)
                f.writerow(["{}_{}".format(l,j),int(out_cost[j].view(-1).detach().cpu().numpy()),int(cost[j].view(-1).detach().cpu().numpy())])

        # Criteria for evaluation
        total_Loss_test += train_loss.item()
        total_evaluation_criteria_test_temp1 = evaluation(out_image1, segment_image1)
        total_evaluation_criteria_test_temp2 = evaluation(out_image2, segment_image2)
        eva1 = [total_evaluation_criteria_test_temp1[k] + eva1[k] for k in
                range(len(eva1))]
        eva2 = [total_evaluation_criteria_test_temp2[k] + eva2[k] for k in
                range(len(eva2))]
        total_evaluation_criteria_test_temp = [
            (total_evaluation_criteria_test_temp1[t] + total_evaluation_criteria_test_temp2[t]) / 2 for t in
            range(min(len(total_evaluation_criteria_test_temp1), len(total_evaluation_criteria_test_temp2)))]
        total_evaluation_criteria_test = [
            total_evaluation_criteria_test[k] + total_evaluation_criteria_test_temp[k] for k in
            range(len(total_evaluation_criteria_test_temp))]
        # if l==10:
        #     break
    loss = total_Loss_test / (len(test_loader))
    #    evaluation_criteria = [recall, precision, acc, fnr, F1, IOU, DICE]

    # loss = total_Loss / (len(test_loader))
    ave_evaluation_criteria_test = [total_evaluation_criteria_test[k] / (len(test_loader)) for k in
                                    range(len(total_evaluation_criteria_test))]
    ave_eva1 = [eva1[k] / (len(test_loader)) for k in range(len(eva1))]
    ave_eva2 = [eva2[k] / (len(test_loader)) for k in range(len(eva2))]
    print(
        "[test result on test 1 dataset]--ave loss: {:.6f}---recall: {:.4f}%, precision: {:.4f}%, acc: {:.4f}%, fnr: {:.4f}%, F1: {:.4f}, IOU: {:.4f}, DICE: {:.4f}".format( \
            loss, 100. * ave_evaluation_criteria_test[0], 100. * ave_evaluation_criteria_test[1],
                  100. * ave_evaluation_criteria_test[2], 100. * ave_evaluation_criteria_test[3], \
            ave_evaluation_criteria_test[4], ave_evaluation_criteria_test[5],
            ave_evaluation_criteria_test[6]))
    print("************************SEG1*****************************")
    print(
        "[test result on test 1 dataset]-----recall: {:.4f}%, precision: {:.4f}%, acc: {:.4f}%, fnr: {:.4f}%, F1: {:.4f}, IOU: {:.4f}, DICE: {:.4f}".format( \
            100. * ave_eva1[0], 100. * ave_eva1[1], 100. * ave_eva1[2], 100. * ave_eva1[3], ave_eva1[4],
            ave_eva1[5],
            ave_eva1[6]))
    print("************************SEG2*****************************")
    print(
        "[test result on test 1 dataset]-----recall: {:.4f}%, precision: {:.4f}%, acc: {:.4f}%, fnr: {:.4f}%, F1: {:.4f}, IOU: {:.4f}, DICE: {:.4f}".format( \
            100. * ave_eva2[0], 100. * ave_eva2[1], 100. * ave_eva2[2], 100. * ave_eva2[3], ave_eva2[4],
            ave_eva2[5],
            ave_eva2[6]))
