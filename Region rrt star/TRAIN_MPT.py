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
from model2 import model2
from model3 import model3
from model4 import model4
from NEED import NEED
from Neural_Driven import *
from Models import *
import dsntnn

#BCE 针对pixel进行平均
#DICE 针对map

'''hyperparameters'''
batch_num = 16
batch_num_validate = 16
batch_num_test = 16
deepSupervision=True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path= r'C:\Users\huang\Desktop\Region RRTStar V2\Params\unet.pth'
data_path='Data'
save_weight_path='Params/unet_%d.pth'



if __name__ == '__main__':
    torch.cuda.empty_cache()
    custom_dataset = MyDataset(data_path)
    train_size = int(len(custom_dataset) * 0.8) #0.8,0.1,0.1
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

    # net = Neural_Driven_pre().to(device)
    net = Transformer(6, 3, 512, 256, 512, 1024, None, 0.1, 40*40, [16, 16]).to('cuda')

    loss_fun1 = nn.BCELoss()
    loss_fun2 = SoftDiceLoss()
    loss_fun3 = nn.MSELoss()
    loss_fun4 = FocalLoss()
    opt = optim.Adam(net.parameters(), lr=0.00003)
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
    alpha = 1
    while epoch < epoch_num:
        since = time.time()
        total_Loss = 0
        total_batch_Loss = 0
        total_evaluation_criteria = [0, 0, 0, 0, 0, 0, 0]
        total_batch_evaluation_criteria = [0, 0, 0, 0, 0, 0, 0]
        net.train()
        for i, (image, segment_image1, segment_image2) in enumerate(data_loader):
            # i : 0~len(data_loader.dataset)/batch_num=len(data_loader)
            # bs=16, 总16000，每次输入16张，共750次循环，每75次输出batch ave loss
            image, segment_image1, segment_image2 = image.to(device), segment_image1.to(device), segment_image2.to(device)

            out_image1 = net(image)
            train_loss1 = loss_fun2(out_image1, segment_image1).to(device)
            train_loss2 = loss_fun1(out_image1, segment_image1).to(device)

            # Per-location euclidean losses

            train_loss = alpha*train_loss1 + train_loss2

            # Criteria for evaluation
            opt.zero_grad()
            train_loss.backward()

            opt.step()
            # Criteria for evaluation
            total_Loss += train_loss.item()
            total_batch_Loss += train_loss.item()

            total_evaluation_criteria_temp = evaluation(out_image1, segment_image1)
            total_evaluation_criteria = [total_evaluation_criteria_temp[k] + total_evaluation_criteria[k] for k in
                                         range(len(total_evaluation_criteria_temp))]
            total_batch_evaluation_criteria_temp = evaluation(out_image1, segment_image1)
            total_batch_evaluation_criteria = [total_batch_evaluation_criteria_temp[k]+total_batch_evaluation_criteria[k] for k in range(len(total_batch_evaluation_criteria_temp))]
               # evaluation_criteria = [recall, precision, acc, fnr, F1, IOU, DICE]

            '''analysis'''
            # data_loader len: samples number in each batch
            # data_loader.dataser len: samplers number totally
            if i == 0:
                print('start epoch: %d' % epoch)
            if math.floor(i % (len(data_loader)/10)) == (len(data_loader)/10)-1:
            # if i!=0:
                print("epoch: {} [{}/{} ({:0f}%)]---loss: {:.4f}---recall: {:.4f}%, precision: {:.4f}%, acc: {:.4f}%, fnr: {:.4f}%, F1: {:.4f}, IOU: {:.4f}, DICE: {:.4f}".format(epoch, (i+1) * len(image), len(data_loader.dataset),100. * (i+1) / len(data_loader), total_batch_Loss/ (len(data_loader)/10),100. *total_batch_evaluation_criteria[0]/(len(data_loader)/10),100. *total_batch_evaluation_criteria[1]/(len(data_loader)/10) \
                                                                                                                                                                                  ,100. *total_batch_evaluation_criteria[2]/(len(data_loader)/10),100. *total_batch_evaluation_criteria[3]/(len(data_loader)/10),total_batch_evaluation_criteria[4]/(len(data_loader)/10), total_batch_evaluation_criteria[5]/(len(data_loader)/10), total_batch_evaluation_criteria[6]/(len(data_loader)/10)))

                total_batch_Loss=0
                total_batch_evaluation_criteria=[0, 0, 0, 0, 0, 0, 0]

            # print('Epoch ' + str(epoch) + ' : ' + str(i // 200) + ' , LOSS =' + str(train_loss.item()))
            if i == len(data_loader) - 1:
                print(
                    "epoch: {}--ave loss: {:.6f}---recall: {:.4f}%, precision: {:.4f}%, acc: {:.4f}%, fnr: {:.4f}%, F1: {:.4f}, IOU: {:.4f}, DICE: {:.4f}".format(
                        epoch, total_Loss / (len(data_loader)), 100. * total_evaluation_criteria[0] / len(data_loader),
                        100. * total_evaluation_criteria[1] / (len(data_loader)),
                        100. * total_evaluation_criteria[2] / (len(data_loader))\
                        , 100. * total_evaluation_criteria[3] / (len(data_loader)),
                        total_evaluation_criteria[4] / (len(data_loader)),
                        total_evaluation_criteria[5] / (len(data_loader)),
                        total_evaluation_criteria[6] / (len(data_loader))))

                #    evaluation_criteria = [recall, precision, acc, fnr, F1, IOU, DICE]
                ave_Loss.append(total_Loss / (len(data_loader)))
                ave_recall.append(total_evaluation_criteria[0] / (len(data_loader)))
                ave_precision.append(total_evaluation_criteria[1] / (len(data_loader)))
                ave_acc.append(total_evaluation_criteria[2] / (len(data_loader)))
                ave_fnr.append(total_evaluation_criteria[3] / (len(data_loader)))
                ave_F1.append(total_evaluation_criteria[4] / (len(data_loader)))
                ave_IOU.append(total_evaluation_criteria[5] / (len(data_loader)))
                ave_DICE.append(total_evaluation_criteria[6] / (len(data_loader)))


            '''save weights'''
            if i == len(data_loader) - 1:
                torch.save(net.state_dict(), save_weight_path % epoch)

        '''test on validate dataset'''
        print('******test on validate dataset******')
        torch.cuda.empty_cache()
        net.eval()
        total_Loss_validate = 0
        total_evaluation_criteria_validate = [0, 0, 0, 0, 0, 0, 0]
        for i, (image, segment_image1, segment_image2) in enumerate(validate_loader):
            # i : 0~len(data_loader.dataset)/batch_num=len(data_loader)
            # bs=16, 总16000，每次输入16张，共750次循环，每75次输出batch ave loss
            image, segment_image1, segment_image2 = image.to(device), segment_image1.to(device), segment_image2.to(device)
            with torch.no_grad():
                out_image1 = net(image)
                train_loss1 = loss_fun2(out_image1, segment_image1).to(device)
                train_loss2 = loss_fun1(out_image1, segment_image1).to(device)

                # Per-location euclidean losses

                train_loss = alpha * train_loss1 + train_loss2

            # Criteria for evaluation
        # Criteria for evaluation
            total_Loss_validate += train_loss.item()

            total_evaluation_criteria_validate_temp = evaluation(out_image1, segment_image1)
            total_evaluation_criteria_validate = [
                total_evaluation_criteria_validate[k] + total_evaluation_criteria_validate_temp[k] for k in
                range(len(total_evaluation_criteria_validate_temp))]

            #    evaluation_criteria = [recall, precision, acc, fnr, F1, IOU, DICE]

        loss = total_Loss_validate / (len(validate_loader))

        ave_evaluation_criteria_validate = [total_evaluation_criteria_validate[k] / (len(validate_loader)) for k in
                                            range(len(total_evaluation_criteria_validate))]
        print(
            "[test result on validate dataset]--ave loss: {:.6f}---recall: {:.4f}%, precision: {:.4f}%, acc: {:.4f}%, fnr: {:.4f}%, F1: {:.4f}, IOU: {:.4f}, DICE: {:.4f}".format( \
                loss, 100. * ave_evaluation_criteria_validate[0], 100. * ave_evaluation_criteria_validate[1],
                      100. * ave_evaluation_criteria_validate[2], 100. * ave_evaluation_criteria_validate[3], \
                ave_evaluation_criteria_validate[4], ave_evaluation_criteria_validate[5],
                ave_evaluation_criteria_validate[6]))
        ave_Loss_validate.append(loss)
        ave_recall_validate.append(ave_evaluation_criteria_validate[0])
        ave_precision_validate.append(ave_evaluation_criteria_validate[1])
        ave_acc_validate.append(ave_evaluation_criteria_validate[2])
        ave_fnr_validate.append(ave_evaluation_criteria_validate[3])
        ave_F1_validate.append(ave_evaluation_criteria_validate[4])
        ave_IOU_validate.append(ave_evaluation_criteria_validate[5])
        ave_DICE_validate.append(ave_evaluation_criteria_validate[6])

        time_elapsed = time.time() - since
        print('Epoch {} training complete in {:.0f}m {:.0f}s'.format(epoch,
                                                                     time_elapsed // 60, time_elapsed % 60))
        epoch += 1  #:D
        print(40*'*')

    # break_epoch = epoch - 1 #:D
    break_epoch = ave_DICE_validate.index(max(ave_DICE_validate)) + epoch_initial
    # break_epoch = 64
    print('the break epoch is {}'.format(break_epoch))

    epoch = epoch - epoch_initial  #:D
    draw_comparision(ave_Loss, ave_Loss_validate, epoch, label1="loss: train dataset", label2="loss: validate dataset",
                     index="loss", title="Comparison Loss")
    draw_comparision(ave_F1, ave_F1_validate, epoch, label1="F1: train dataset", label2="F1: validate dataset",
                     index="F1", title="Comparison F1")
    draw_comparision(ave_IOU, ave_IOU_validate, epoch, label1="IOU: train dataset", label2="IOU: validate dataset",
                     index="IOU", title="Comparison IOU")
    draw_comparision(ave_DICE, ave_DICE_validate, epoch, label1="DICE: train dataset", label2="DICE: validate dataset",
                     index="DICE", title="Comparison Dice")

    '''draw figure[training procedure]'''
    plot_figure(ave_Loss, epoch, title='loss-epoch[train dataset]', x_label='epoch', y_label='loss')
    plot_figure(ave_recall, epoch, title='recall-epoch[train dataset]', x_label='epoch', y_label='recall')
    plot_figure(ave_precision, epoch, title='precision-epoch[train dataset]', x_label='epoch', y_label='precision')
    plot_figure(ave_acc, epoch, title='accuracy-epoch[train dataset]', x_label='epoch', y_label='accuracy')
    plot_figure(ave_fnr, epoch, title='false negative rate(fnr)-epoch[train dataset]', x_label='epoch',
                y_label='false negative rate(fnr)')
    plot_figure(ave_F1, epoch, title='F1 score-epoch[train dataset]', x_label='epoch', y_label='F1 score')
    plot_figure(ave_IOU, epoch, title='IOU-epoch[train dataset]', x_label='epoch', y_label='IOU')
    plot_figure(ave_DICE, epoch, title='DICE-epoch[train dataset]', x_label='epoch', y_label='DICE')
    # plot_loss_acc(ave_Loss, ave_acc, epoch_num)
    '''draw figure[validate procedure]'''
    plot_figure(ave_Loss_validate, epoch, title='loss-epoch[validate dataset]', x_label='epoch', y_label='loss')
    plot_figure(ave_recall_validate, epoch, title='recall-epoch[validate dataset]', x_label='epoch', y_label='recall')
    plot_figure(ave_precision_validate, epoch, title='precision-epoch[validate dataset]', x_label='epoch',
                y_label='precision')
    plot_figure(ave_acc_validate, epoch, title='accuracy-epoch[validate dataset]', x_label='epoch', y_label='accuracy')
    plot_figure(ave_fnr_validate, epoch, title='false negative rate(fnr)-epoch[validate dataset]', x_label='epoch',
                y_label='false negative rate(fnr)')
    plot_figure(ave_F1_validate, epoch, title='F1 score-epoch[validate dataset]', x_label='epoch', y_label='F1 score')
    plot_figure(ave_IOU_validate, epoch, title='IOU-epoch[validate dataset]', x_label='epoch', y_label='IOU')
    plot_figure(ave_DICE_validate, epoch, title='DICE-epoch[validate dataset]', x_label='epoch', y_label='DICE')

    '''test 1 on test dataset'''
    # print("the best epoch: {}".format(break_epoch))
    print('******test 1 on test dataset******')
    torch.cuda.empty_cache()
    net.eval()
    with open(F'CSVFile/cost_result1.csv', 'w') as csvfile:
        f = csv.writer(csvfile)
        f.writerow(["Index", "Coordinate_PRE", "Coordinate_GT"])
    total_Loss_test = 0
    total_Loss_test_reg = 0
    total_evaluation_criteria_test = [0, 0, 0, 0, 0, 0, 0]
    ave_evaluation_criteria_test = [0, 0, 0, 0, 0, 0, 0]
    weights = save_weight_path%break_epoch
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successfully load weights')
    else:
        print('unsuccessfully load weights')
    for l, (image, segment_image1, segment_image2) in enumerate(test_loader):
        # i : 0~len(data_loader.dataset)/batch_num=len(data_loader)
        # bs=16, 总16000，每次输入16张，共750次循环，每75次输出batch ave loss
        image, segment_image1, segment_image2 = image.to(device), segment_image1.to(device), segment_image2.to(device)
        with torch.no_grad():
            out_image1 = net(image)
            train_loss1 = loss_fun2(out_image1, segment_image1).to(device)
            train_loss2 = loss_fun1(out_image1, segment_image1).to(device)

            # Per-location euclidean losses

            train_loss = alpha*train_loss1 + train_loss2
        total_Loss_test += train_loss.item()

        # Criteria for evaluation

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

            # with open(F'CSVFile/cost_result1.csv', 'a', newline='') as csvfile:
            #     f = csv.writer(csvfile)
            #     f.writerow(["{}_{}".format(l, j), np.round(torch.reshape(out_coord[j], (-1,)).detach().cpu().numpy(),0),
            #                 np.round(torch.reshape(coord[j,:,0:2],(-1,)).detach().cpu().numpy(),4)])
        total_evaluation_criteria_test_temp = evaluation(out_image1, segment_image1)
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
    print(
        "[test result on test 1 dataset]--ave loss: {:.6f}---recall: {:.4f}%, precision: {:.4f}%, acc: {:.4f}%, fnr: {:.4f}%, F1: {:.4f}, IOU: {:.4f}, DICE: {:.4f}".format( \
            loss, 100. * ave_evaluation_criteria_test[0], 100. * ave_evaluation_criteria_test[1],
                  100. * ave_evaluation_criteria_test[2], 100. * ave_evaluation_criteria_test[3], \
            ave_evaluation_criteria_test[4], ave_evaluation_criteria_test[5],
            ave_evaluation_criteria_test[6]))


    '''test 2 on test dataset'''
    print('******test 2 on designed test dataset******')
    torch.cuda.empty_cache()
    net.eval()
    with open(F'CSVFile/cost_result2.csv','w') as csvfile:
        f = csv.writer(csvfile)
        f.writerow(["Index", "Coordinate_PRE", "Coordinate_GT"])
    test_2_data_path = r'C:\Users\huang\Desktop\Region RRTStar V2\Data\test1'
    SEENdataloader = DataLoader(MyDataset(test_2_data_path), batch_size=16, shuffle=False)

    total_Loss_test = 0
    total_evaluation_criteria_test = [0, 0, 0, 0, 0, 0, 0]
    ave_evaluation_criteria_test = [0, 0, 0, 0, 0, 0, 0]

    weights = save_weight_path%break_epoch
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successfully load weights')
    else:
        print('unsuccessfully load weights')
    for l, (image, segment_image1, segment_image2) in enumerate(SEENdataloader):
        # i : 0~len(data_loader.dataset)/batch_num=len(data_loader)
        # bs=16, 总16000，每次输入16张，共750次循环，每75次输出batch ave loss
        image, segment_image1, segment_image2 = image.to(device), segment_image1.to(device), segment_image2.to(device)
        with torch.no_grad():

            out_image1 = net(image)
            train_loss1 = loss_fun2(out_image1, segment_image1).to(device)
            train_loss2 = loss_fun1(out_image1, segment_image1).to(device)

            # Per-location euclidean losses

            train_loss = alpha*train_loss1 + train_loss2
        # Criteria for evaluation
        total_Loss_test += train_loss.item()

        for j in range(batch_num_test):
            img = torch.squeeze(image[j])
            predict_img = torch.squeeze(out_image1[j])
            predict_img1 = torch.stack([predict_img, predict_img], dim=0)
            true_img = torch.squeeze(segment_image1[j])
            true_img1 = torch.stack([true_img, true_img], dim=0)

            save_image_tensor2pillow(predict_img1, F'TestResult/Result2/1_prediction{l}_{j}.jpg')
            save_image_tensor2pillow(true_img1, F'TestResult/Result2/1_groundtruth{l}_{j}.jpg')

            original_img = image[j]
            unit_image1 = torch.stack([predict_img, true_img], dim=0)
            unit_image2 = torch.stack([original_img[0] - 110 / 255 * torch.squeeze(predict_img),
                                       original_img[1] - torch.squeeze(predict_img),
                                       original_img[2] - torch.squeeze(predict_img)], dim=0)
            unit_image2 = torch.squeeze(unit_image2)

            save_image_tensor2pillow(unit_image2, F'TestResult/Result2/1_{l}_{j}_original.jpg')
            save_image_tensor2pillow(unit_image1, F'TestResult/Result2/1_{l}_{j}.jpg')


            # with open(F'CSVFile/cost_result2.csv', 'a', newline='') as csvfile:
            #     f = csv.writer(csvfile)
            #     f.writerow(["{}_{}".format(l, j), np.round(torch.reshape(out_coord[j], (-1,)).detach().cpu().numpy(),0),
            #                 np.round(torch.reshape(coord[j,:,0:2],(-1,)).detach().cpu().numpy(),4)])
        total_evaluation_criteria_test_temp = evaluation(out_image1, segment_image1)

        total_evaluation_criteria_test = [
            total_evaluation_criteria_test[k] + total_evaluation_criteria_test_temp[k] for k in
            range(len(total_evaluation_criteria_test_temp))]

    loss = total_Loss_test / (len(SEENdataloader))

    #    evaluation_criteria = [recall, precision, acc, fnr, F1, IOU, DICE]

    ave_evaluation_criteria_test = [total_evaluation_criteria_test[k] / (len(SEENdataloader)) for k in
                                    range(len(total_evaluation_criteria_test))]
    print(
        "[test result on test 2 dataset]--ave loss: {:.6f}---recall: {:.4f}%, precision: {:.4f}%, acc: {:.4f}%, fnr: {:.4f}%, F1: {:.4f}, IOU: {:.4f}, DICE: {:.4f}".format( \
            loss, 100. * ave_evaluation_criteria_test[0], 100. * ave_evaluation_criteria_test[1],
                  100. * ave_evaluation_criteria_test[2], 100. * ave_evaluation_criteria_test[3], \
            ave_evaluation_criteria_test[4], ave_evaluation_criteria_test[5],
            ave_evaluation_criteria_test[6]))

    '''test 3 on test dataset'''
    print('******test 3 on designed test dataset******')
    torch.cuda.empty_cache()
    with open(F'CSVFile/cost_result3.csv', 'w') as csvfile:
        f = csv.writer(csvfile)
        f.writerow(["Index", "Coordinate_PRE", "Coordinate_GT"])
    test_3_data_path = r'C:\Users\huang\Desktop\Region RRTStar V2\Data\test2'
    UNSEENdataloader = DataLoader(MyDataset(test_3_data_path), batch_size=16, shuffle=False)

    total_Loss_test = 0
    total_evaluation_criteria_test = [0, 0, 0, 0, 0, 0, 0]
    ave_evaluation_criteria_test = [0, 0, 0, 0, 0, 0, 0]

    weights = save_weight_path%break_epoch
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successfully load weights')
    else:
        print('unsuccessfully load weights')
    for l, (image, segment_image1, segment_image2) in enumerate(UNSEENdataloader):
        # i : 0~len(data_loader.dataset)/batch_num=len(data_loader)
        # bs=16, 总16000，每次输入16张，共750次循环，每75次输出batch ave loss
        image, segment_image1, segment_image2 = image.to(device), segment_image1.to(device), segment_image2.to(device)
        with torch.no_grad():

            out_image1 = net(image)
            train_loss1 = loss_fun2(out_image1, segment_image1).to(device)
            train_loss2 = loss_fun1(out_image1, segment_image1).to(device)

            # Per-location euclidean losses

            train_loss = alpha*train_loss1 + train_loss2
        # Criteria for evaluation
        total_Loss_test += train_loss.item()

        for j in range(batch_num_test):
            img = torch.squeeze(image[j])
            predict_img = torch.squeeze(out_image1[j])
            predict_img1 = torch.stack([predict_img, predict_img], dim=0)
            true_img = torch.squeeze(segment_image1[j])
            true_img1 = torch.stack([true_img, true_img], dim=0)

            save_image_tensor2pillow(predict_img1, F'TestResult/Result3/1_prediction{l}_{j}.jpg')
            save_image_tensor2pillow(true_img1, F'TestResult/Result3/1_groundtruth{l}_{j}.jpg')

            original_img = image[j]
            unit_image1 = torch.stack([predict_img, true_img], dim=0)
            unit_image2 = torch.stack([original_img[0] - 110 / 255 * torch.squeeze(predict_img),
                                       original_img[1] - torch.squeeze(predict_img),
                                       original_img[2] - torch.squeeze(predict_img)], dim=0)
            unit_image2 = torch.squeeze(unit_image2)

            save_image_tensor2pillow(unit_image2, F'TestResult/Result3/1_{l}_{j}_original.jpg')
            save_image_tensor2pillow(unit_image1, F'TestResult/Result3/1_{l}_{j}.jpg')



            # with open(F'CSVFile/cost_result3.csv', 'a', newline='') as csvfile:
            #     f = csv.writer(csvfile)
            #     f.writerow(["{}_{}".format(l, j), np.round(torch.reshape(out_coord[j], (-1,)).detach().cpu().numpy(),0),
            #                 np.round(torch.reshape(coord[j,:,0:2],(-1,)).detach().cpu().numpy(),4)])
        total_evaluation_criteria_test_temp = evaluation(out_image1, segment_image1)
        total_evaluation_criteria_test = [total_evaluation_criteria_test[k] + total_evaluation_criteria_test_temp[k] for k in
                range(len(total_evaluation_criteria_test_temp))]

    loss = total_Loss_test / (len(UNSEENdataloader))

        #    evaluation_criteria = [recall, precision, acc, fnr, F1, IOU, DICE]
    ave_evaluation_criteria_test = [total_evaluation_criteria_test[k] / (len(UNSEENdataloader)) for k in
                                        range(len(total_evaluation_criteria_test))]
    print(
            "[test result on test 3 dataset]--ave loss: {:.6f}---recall: {:.4f}%, precision: {:.4f}%, acc: {:.4f}%, fnr: {:.4f}%, F1: {:.4f}, IOU: {:.4f}, DICE: {:.4f}".format( \
                loss, 100. * ave_evaluation_criteria_test[0], 100. * ave_evaluation_criteria_test[1],
                      100. * ave_evaluation_criteria_test[2], 100. * ave_evaluation_criteria_test[3], \
                ave_evaluation_criteria_test[4], ave_evaluation_criteria_test[5],
                ave_evaluation_criteria_test[6]))
