import csv
import random
import time

from torch import nn,optim
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from torch.utils.data import DataLoader
import torch.utils.data
from PIL import Image
from data import *
# from net import *
from Customed_Loss import *
from utils import *
from torchvision.utils import save_image
from model1 import model1

#BCE 针对pixel进行平均
#DICE 针对map

'''hyperparameters'''
batch_num = 16
batch_num_validate = 16
batch_num_test = 16
deepSupervision=True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path= r'C:\Users\huang\Desktop\Region RRTStar V2\Params\unet_%d.pth'
data_path='Data'



if __name__ == '__main__':
    torch.cuda.empty_cache()
    custom_dataset = MyDataset(data_path)
    train_size = int(len(custom_dataset) * 0.8) #0.75:0.125:0.125
    validate_size = int(len(custom_dataset) * 0.1)
    test_size = int(len(custom_dataset))-train_size-validate_size
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(custom_dataset,[train_size, validate_size, test_size], generator=torch.Generator().manual_seed(0))
    # train_dataset, validate_dataset = torch.utils.data.random_split(custom_dataset,[train_size, validate_size], generator=torch.Generator().manual_seed(0))


    data_loader = DataLoader(train_dataset, batch_size=batch_num, shuffle=True, num_workers=8, pin_memory=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_num_validate, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_num_test, shuffle=False, num_workers=8, pin_memory=True)
    seed=42
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    net=model1().to(device)
    loss_fun1 = nn.BCELoss()
    loss_fun2 = SoftDiceLoss()
    loss_fun3 = nn.MSELoss()

    opt = optim.Adam(net.parameters(), lr=0.0003)
    # schedular = CosineAnnealingWarmRestarts(opt,T_0=1,T_mult=2)
    net.train()
    print('# generator parameters:', 1.0 * sum(param.numel() for param in net.parameters()) / 1000000)

    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successfully load weight')
    else:
        print('unsuccessfully load weight')

    epoch = 1
    epoch_initial = epoch
    epoch_num = 33
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

    '''test 1 on test dataset'''
    # print("the best epoch: {}".format(break_epoch))
    print('******test 1 on test dataset******')
    torch.cuda.empty_cache()
    net.eval()
    with open(F'CSVFile/cost_result1.csv', 'w') as csvfile:
        f = csv.writer(csvfile)
        f.writerow(["index", "prediction", "ground truth"])
    total_Loss_test = 0
    total_Loss_test_reg = 0
    total_evaluation_criteria_test = [0, 0, 0, 0, 0, 0, 0]
    ave_evaluation_criteria_test = [0, 0, 0, 0, 0, 0, 0]
    break_epoch = int(input("type in the best epoch:"))
    weights = weight_path%break_epoch
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successfully load weights')
    else:
        print('unsuccessfully load weights')
    for l, (image, segment_image, coord) in enumerate(test_loader):
        # i : 0~len(data_loader.dataset)/batch_num=len(data_loader)
        # bs=16, 总16000，每次输入16张，共750次循环，每75次输出batch ave loss
        image, segment_image, coord = image.to(device), segment_image.to(device), coord.to(device)
        with torch.no_grad():
            out_image, out_coord = net(image)
            train_loss1 = loss_fun1(out_image, segment_image).to(device)
            train_loss2 = loss_fun2(out_image, segment_image).to(device)
            a = out_coord
            b = torch.reshape(coord[:,:,0:2],(-1,40))
            train_loss3 = loss_fun3(a, b).to(device)/1000
            train_loss = train_loss1 + train_loss2 + train_loss3
            train_loss_reg = sqrt(train_loss3*1000)
        # Criteria for evaluation
        total_Loss_test_reg += train_loss_reg
        total_Loss_test += train_loss.item()
        for j in range(batch_num_test):
            img = torch.squeeze(image[j])
            predict_img = torch.squeeze(out_image[j])
            predict_img1 = torch.stack([predict_img, predict_img], dim=0)
            true_img = torch.squeeze(segment_image[j])
            true_img1 = torch.stack([true_img, true_img], dim=0)

            save_image_tensor2pillow(img, F'TestResult/Result1/map{l}_{j}.jpg')
            save_image_tensor2pillow(predict_img1, F'TestResult/Result1/prediction{l}_{j}.jpg')
            save_image_tensor2pillow(true_img1, F'TestResult/Result1/groundtruth{l}_{j}.jpg')

            original_img = image[j]
            unit_image1 = torch.stack([predict_img, true_img], dim=0)
            unit_image2 = torch.stack([original_img[0] - 110 / 255 * torch.squeeze(predict_img),
                                       original_img[1] - torch.squeeze(predict_img),
                                       original_img[2] - torch.squeeze(predict_img)], dim=0)
            unit_image2 = torch.squeeze(unit_image2)

            save_image_tensor2pillow(unit_image2, F'TestResult/Result1/{l}_{j}_original.jpg')
            save_image_tensor2pillow(unit_image1, F'TestResult/Result1/{l}_{j}.jpg')
            with open(F'CSVFile/cost_result1.csv', 'a', newline='') as csvfile:
                f = csv.writer(csvfile)
                f.writerow(["{}_{}".format(l, j), np.round(torch.reshape(out_coord[j], (-1,)).detach().cpu().numpy(),0),
                            np.round(torch.reshape(coord[j,:,0:2],(-1,)).detach().cpu().numpy(),4)])
        total_evaluation_criteria_test_temp = evaluation(out_image, segment_image)
        total_evaluation_criteria_test = [
            total_evaluation_criteria_test[k] + total_evaluation_criteria_test_temp[k] for k in
            range(len(total_evaluation_criteria_test_temp))]
        # if l==10:
        #     break
    loss = total_Loss_test / (len(test_loader))
    loss_reg = total_Loss_test_reg/(len(test_loader))
    #    evaluation_criteria = [recall, precision, acc, fnr, F1, IOU, DICE]


    # loss = total_Loss / (len(test_loader))
    ave_evaluation_criteria_test = [total_evaluation_criteria_test[k] / (len(test_loader)) for k in
                                    range(len(total_evaluation_criteria_test))]
    print(
        "[test result on test 1 dataset]--ave loss: {:.6f}---recall: {:.4f}%, precision: {:.4f}%, acc: {:.4f}%, fnr: {:.4f}%, F1: {:.4f}, IOU: {:.4f}, DICE: {:.4f}".format( \
            loss, 100. * ave_evaluation_criteria_test[0], 100. * ave_evaluation_criteria_test[1],
                  100. * ave_evaluation_criteria_test[2], 100. * ave_evaluation_criteria_test[3], \
            ave_evaluation_criteria_test[4], ave_evaluation_criteria_test[5],
            ave_evaluation_criteria_test[6]))
    print("Regression Result:{:.4f}".format(loss_reg))


    '''test 2 on test dataset'''
    print('******test 2 on designed test dataset******')
    torch.cuda.empty_cache()
    net.eval()
    with open(F'CSVFile/cost_result2.csv','w') as csvfile:
        f = csv.writer(csvfile)
        f.writerow(["index","prediction","ground truth"])
    test_2_data_path = r'C:\Users\huang\Desktop\Region RRTStar V2\Data\test1'
    SEENdataloader = DataLoader(MyDataset(test_2_data_path), batch_size=16, shuffle=False)

    total_Loss_test = 0
    total_Loss_test_reg = 0
    total_evaluation_criteria_test = [0, 0, 0, 0, 0, 0, 0]
    ave_evaluation_criteria_test = [0, 0, 0, 0, 0, 0, 0]

    weights = weight_path%break_epoch
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successfully load weights')
    else:
        print('unsuccessfully load weights')
    for l, (image, segment_image, coord) in enumerate(SEENdataloader):
        # i : 0~len(data_loader.dataset)/batch_num=len(data_loader)
        # bs=16, 总16000，每次输入16张，共750次循环，每75次输出batch ave loss
        image, segment_image, coord = image.to(device), segment_image.to(device), coord.to(device)
        with torch.no_grad():
            out_image, out_coord = net(image)
            train_loss1 = loss_fun1(out_image, segment_image).to(device)
            train_loss2 = loss_fun2(out_image, segment_image).to(device)
            a = out_coord
            b = torch.reshape(coord[:, :, 0:2], (-1, 40))
            train_loss3 = loss_fun3(a, b).to(device)/1000
            train_loss = train_loss1 + train_loss2 + train_loss3
            train_loss_reg = sqrt(train_loss3*1000)
        # Criteria for evaluation
        total_Loss_test_reg += train_loss_reg
        total_Loss_test += train_loss.item()

        for j in range(batch_num_test):
            img = torch.squeeze(image[j])
            predict_img = torch.squeeze(out_image[j])
            predict_img1 = torch.stack([predict_img, predict_img], dim=0)
            true_img = torch.squeeze(segment_image[j])
            true_img1 = torch.stack([true_img, true_img], dim=0)
            save_image_tensor2pillow(img, F'TestResult/Result2/map{l}_{j}.jpg')
            save_image_tensor2pillow(predict_img1, F'TestResult/Result2/prediction{l}_{j}.jpg')
            save_image_tensor2pillow(true_img1, F'TestResult/Result2/groundtruth{l}_{j}.jpg')

            original_img = image[j]
            unit_image1 = torch.stack([predict_img, true_img], dim=0)
            unit_image2 = torch.stack([original_img[0] - 110 / 255 * torch.squeeze(predict_img),
                                       original_img[1] - torch.squeeze(predict_img),
                                       original_img[2] - torch.squeeze(predict_img)], dim=0)
            unit_image2 = torch.squeeze(unit_image2)
            save_image_tensor2pillow(unit_image2, F'TestResult/Result2/{l}_{j}_original.jpg')
            save_image_tensor2pillow(unit_image1, F'TestResult/Result2/{l}_{j}.jpg')
            with open(F'CSVFile/cost_result2.csv', 'a', newline='') as csvfile:
                f = csv.writer(csvfile)
                f.writerow(["{}_{}".format(l, j), np.round(torch.reshape(out_coord[j], (-1,)).detach().cpu().numpy(),0),
                            np.round(torch.reshape(coord[j,:,0:2],(-1,)).detach().cpu().numpy(),4)])
        total_evaluation_criteria_test_temp = evaluation(out_image, segment_image)
        total_evaluation_criteria_test = [
            total_evaluation_criteria_test[k] + total_evaluation_criteria_test_temp[k] for k in
            range(len(total_evaluation_criteria_test_temp))]

    loss = total_Loss_test / (len(SEENdataloader))
    loss_reg = total_Loss_test_reg/(len(SEENdataloader))

    #    evaluation_criteria = [recall, precision, acc, fnr, F1, IOU, DICE]
    ave_evaluation_criteria_test = [total_evaluation_criteria_test[k] / (len(test_loader)) for k in
                                    range(len(total_evaluation_criteria_test))]
    print(
        "[test result on test 2 dataset]--ave loss: {:.6f}---recall: {:.4f}%, precision: {:.4f}%, acc: {:.4f}%, fnr: {:.4f}%, F1: {:.4f}, IOU: {:.4f}, DICE: {:.4f}".format( \
            loss, 100. * ave_evaluation_criteria_test[0], 100. * ave_evaluation_criteria_test[1],
                  100. * ave_evaluation_criteria_test[2], 100. * ave_evaluation_criteria_test[3], \
            ave_evaluation_criteria_test[4], ave_evaluation_criteria_test[5],
            ave_evaluation_criteria_test[6]))
    print("Regression Result:{:.4f}".format(loss_reg))

    '''test 3 on test dataset'''
    print('******test 3 on designed test dataset******')
    torch.cuda.empty_cache()
    with open(F'CSVFile/cost_result3.csv', 'w') as csvfile:
        f = csv.writer(csvfile)
        f.writerow(["index", "prediction", "ground truth"])
    test_3_data_path = r'C:\Users\huang\Desktop\Region RRTStar V2\Data\test2'
    UNSEENdataloader = DataLoader(MyDataset(test_3_data_path), batch_size=16, shuffle=False)

    total_Loss_test = 0
    total_Loss_test_reg = 0
    total_evaluation_criteria_test = [0, 0, 0, 0, 0, 0, 0]
    ave_evaluation_criteria_test = [0, 0, 0, 0, 0, 0, 0]

    weights = weight_path%break_epoch
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successfully load weights')
    else:
        print('unsuccessfully load weights')
    for l, (image, segment_image, coord) in enumerate(UNSEENdataloader):
        # i : 0~len(data_loader.dataset)/batch_num=len(data_loader)
        # bs=16, 总16000，每次输入16张，共750次循环，每75次输出batch ave loss
        image, segment_image, coord = image.to(device), segment_image.to(device), coord.to(device)
        with torch.no_grad():
            out_image, out_coord = net(image)
            train_loss1 = loss_fun1(out_image, segment_image).to(device)
            train_loss2 = loss_fun2(out_image, segment_image).to(device)
            a = out_coord
            b = torch.reshape(coord[:, :, 0:2], (-1, 40))
            train_loss3 = loss_fun3(a, b).to(device)/1000
            train_loss = train_loss1 + train_loss2 + train_loss3
            train_loss_reg = sqrt(train_loss3*1000)
        # Criteria for evaluation
        total_Loss_test_reg += train_loss_reg
        total_Loss_test += train_loss.item()

        for j in range(batch_num_test):
            img = torch.squeeze(image[j])
            predict_img = torch.squeeze(out_image[j])
            predict_img1 = torch.stack([predict_img, predict_img], dim=0)
            true_img = torch.squeeze(segment_image[j])
            true_img1 = torch.stack([true_img, true_img], dim=0)
            save_image_tensor2pillow(img, F'TestResult/Result3/map{l}_{j}.jpg')
            save_image_tensor2pillow(predict_img1, F'TestResult/Result3/prediction{l}_{j}.jpg')
            save_image_tensor2pillow(true_img1, F'TestResult/Result3/groundtruth{l}_{j}.jpg')

            original_img = image[j]
            unit_image1 = torch.stack([predict_img, true_img], dim=0)
            unit_image2 = torch.stack([original_img[0] - 110 / 255 * torch.squeeze(predict_img),
                                       original_img[1] - torch.squeeze(predict_img),
                                       original_img[2] - torch.squeeze(predict_img)], dim=0)
            unit_image2 = torch.squeeze(unit_image2)
            save_image_tensor2pillow(unit_image2, F'TestResult/Result3/{l}_{j}_original.jpg')
            save_image_tensor2pillow(unit_image1, F'TestResult/Result3/{l}_{j}.jpg')
            with open(F'CSVFile/cost_result3.csv', 'a', newline='') as csvfile:
                f = csv.writer(csvfile)
                f.writerow(["{}_{}".format(l, j), np.round(torch.reshape(out_coord[j], (-1,)).detach().cpu().numpy(), 0),
                     np.round(torch.reshape(coord[j, :, 0:2], (-1,)).detach().cpu().numpy(), 4)])
        total_evaluation_criteria_test_temp = evaluation(out_image, segment_image)
        total_evaluation_criteria_test = [total_evaluation_criteria_test[k] + total_evaluation_criteria_test_temp[k] for k in
                range(len(total_evaluation_criteria_test_temp))]

    loss = total_Loss_test / (len(UNSEENdataloader))
    loss_reg = total_Loss_test_reg/(len(UNSEENdataloader))

        #    evaluation_criteria = [recall, precision, acc, fnr, F1, IOU, DICE]
    ave_evaluation_criteria_test = [total_evaluation_criteria_test[k] / (len(test_loader)) for k in
                                        range(len(total_evaluation_criteria_test))]
    print(
            "[test result on test 3 dataset]--ave loss: {:.6f}---recall: {:.4f}%, precision: {:.4f}%, acc: {:.4f}%, fnr: {:.4f}%, F1: {:.4f}, IOU: {:.4f}, DICE: {:.4f}".format( \
                loss, 100. * ave_evaluation_criteria_test[0], 100. * ave_evaluation_criteria_test[1],
                      100. * ave_evaluation_criteria_test[2], 100. * ave_evaluation_criteria_test[3], \
                ave_evaluation_criteria_test[4], ave_evaluation_criteria_test[5],
                ave_evaluation_criteria_test[6]))
    print("Regression Result:{:.4f}".format(loss_reg))
