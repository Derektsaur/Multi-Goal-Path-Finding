import csv
import random
import time
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data import *
from utils_2 import *
from Custom_Loss import *
from Models import *
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device on :{device}')

'''hyperparameters'''
batch_num = 4
batch_num_validate = batch_num
batch_num_test = batch_num
epoch = 1
epoch_num = 11
warmup_epoch = 6
model_args = dict(
    n_layers=10,
    n_heads=8,
    d_k=256,
    d_v=128,
    d_model=256,
    d_inner=512,
    pad_idx=None,
    n_position=16 * 16,
    dropout=0.2,
    train_shape=[16, 16],
)
net = Transformer(**model_args).to(device)
if epoch_num == 3:
    data_path = 'Data_TSP\_Data_debug'
else:
    data_path = 'Data_TSP'
weights_pre = r'best_result/mpt_1020.pth'
max_reg_loss = 1
max_prom_loss = 1
max_line_loss = 1
save_weight_path = 'params\mpt_%d.pth'
test_image_path = 'test_image_mine'
txt_path = 'figure_result'

if __name__ == '__main__':
    torch.cuda.empty_cache()
    cd = MyDataset(data_path)
    ts = int(len(cd) * 0.8)
    vs = int(len(cd) * 0.1)
    test = int(len(cd)) - ts - vs
    td, vd, ttd = torch.utils.data.random_split(cd, [ts, vs, test], generator=torch.Generator().manual_seed(0))
    data_loader = DataLoader(td, batch_size=batch_num, shuffle=True, pin_memory=True)
    validate_loader = DataLoader(vd, batch_size=batch_num_validate, shuffle=False, pin_memory=True)
    test_loader = DataLoader(ttd, batch_size=batch_num_test, shuffle=False, pin_memory=True)
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.empty_cache()
    if os.path.exists(weights_pre):
        try:
            net.load_state_dict(torch.load(weights_pre))
            print(f'successfully load weights [{weights_pre}]')
        except RuntimeError as e:
            print('Error:', e)
    else:
        print('No pretrained weights path')
    opt = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-5)
    Focal_loss = FocalLoss(alpha=0.25, gamma=2).to(device)
    SoftDice_loss = SoftDiceLoss().to(device)
    MSE_loss = nn.MSELoss(reduction="mean").to(device)
    BCE_loss = nn.BCELoss().to(device)
    focal_loss2 = FocalLoss(alpha=2, gamma=0).to(device)
    ave_LISTs = lr_list, ave_Loss, ave_recall, ave_precision, ave_acc, ave_fnr, ave_F1, ave_IOU, ave_DICE, ave_Loss_validate, ave_recall_validate, ave_precision_validate, ave_acc_validate, ave_fnr_validate, ave_F1_validate, ave_IOU_validate, ave_DICE_validate = [
        [] for _ in range(17)]
    ave_NAMEs = ['learning rate', 'ave_Loss', 'ave_recall', 'ave_precision', 'ave_acc', 'ave_fnr', 'ave_F1', 'ave_IOU', 'ave_DICE', 'ave_Loss_validate', 'ave_recall_validate', 'ave_precision_validate', 'ave_acc_validate', 'ave_fnr_validate', 'ave_F1_validate', 'ave_IOU_validate', 'ave_DICE_validate']
    alpha = 2
    st = time.time()
    logger = Logger(txt_path)

    while epoch < epoch_num:
        '''Train'''
        since = time.time()
        torch.cuda.empty_cache()
        net.train()
        R_Loss = 0
        P_Loss = 0
        L_Loss = 0
        metrics1 = [0, 0, 0, 0, 0, 0, 0]
        metrics2 = [0, 0, 0, 0, 0, 0, 0]
        for (image, seg1, seg2, cost, coord) in tqdm(data_loader, desc='Training', colour='blue'):
            image, seg1, seg2, cost = image.to(device), seg1.to(device), seg2.to(device), cost.to(device)
            prom, line, out_cost = net(image)
            reg_loss = MSE_loss(torch.squeeze(out_cost, dim=1), cost)
            prom_loss = Focal_loss(prom, seg1) + SoftDice_loss(prom, seg1)
            line_loss = Focal_loss(line, seg2)
            scaled_reg_loss = reg_loss / max_reg_loss
            scaled_prom_loss = prom_loss / max_prom_loss
            scaled_line_loss = line_loss / max_line_loss
            train_scaled_loss = scaled_reg_loss + scaled_prom_loss + scaled_line_loss
            opt.zero_grad()
            train_scaled_loss.backward()
            opt.step()
            R_Loss += reg_loss.item()
            P_Loss += prom_loss.item()
            L_Loss += line_loss.item()
            recall, precision, acc, fnr, F1, IOU, DICE = evaluation(prom, seg1)
            metrics1 = [m + r.item() for m, r in zip(metrics1, [recall, precision, acc, fnr, F1, IOU, DICE])]
            recall, precision, acc, fnr, F1, IOU, DICE = evaluation(line, seg2)
            metrics2 = [m + r.item() for m, r in zip(metrics2, [recall, precision, acc, fnr, F1, IOU, DICE])]
        train_time = time.time() - since
        r_loss = R_Loss / (len(data_loader))
        p_loss = P_Loss / (len(data_loader))
        l_loss = L_Loss / (len(data_loader))
        metrics1 = [m / (len(data_loader)) for m in metrics1]
        metrics2 = [m / (len(data_loader)) for m in metrics2]
        logger.info(
            f"Epoch {int(epoch)}/{int(epoch_num - 1)}, lr {opt.param_groups[0]['lr']:.6f} ------------------------------")
        logger.info(f"train: time {(train_time // 60):.0f}m {(train_time % 60):.0f}s, regression loss:{r_loss:.4f}, "
                    f"promising loss:{p_loss:.4f}, guideline loss:{l_loss:.4f}")
        lr_list.append(opt.param_groups[0]['lr'])
        ave_Loss.append(r_loss)
        torch.save(net.state_dict(), save_weight_path % epoch)

        '''Eval'''
        torch.cuda.empty_cache()
        net.eval()
        val_start = time.time()
        Loss_validate = 0
        metrics1 = [0, 0, 0, 0, 0, 0, 0]
        metrics2 = [0, 0, 0, 0, 0, 0, 0]
        for (image, seg1, seg2, cost, coord) in tqdm(validate_loader, desc='Validating', colour='MAGENTA'):
            image, seg1, seg2, cost = image.to(device), seg1.to(device), seg2.to(device), cost.to(device)
            with torch.no_grad():
                prom, line, out_cost = net(image)
                train_loss = MSE_loss(torch.squeeze(out_cost, dim=1), cost)
            Loss_validate += train_loss.item()
            recall, precision, acc, fnr, F1, IOU, DICE = evaluation(prom, seg1)
            metrics1 = [m + r.item() for m, r in zip(metrics1, [recall, precision, acc, fnr, F1, IOU, DICE])]
            recall, precision, acc, fnr, F1, IOU, DICE = evaluation(line, seg2)
            metrics2 = [m + r.item() for m, r in zip(metrics2, [recall, precision, acc, fnr, F1, IOU, DICE])]
        val_time = time.time() - val_start
        loss = Loss_validate / (len(validate_loader))
        metrics1 = [m / (len(validate_loader)) for m in metrics1]
        metrics2 = [m / (len(validate_loader)) for m in metrics2]
        logger.info(f"val: time {(val_time // 60):.0f}m {(val_time % 60):.0f}s, regression loss {loss:.4f}\n"
                    f"Seg1 : recall {100. * metrics1[0]:.4f}%, precision {100. * metrics1[1]:.4f}% "
                    f"acc {100. * metrics1[2]:.4f}%, fnr {100. * metrics1[3]:.4f}%, F1 {metrics1[4]:.4f},  "
                    f"IOU {metrics1[5]:.4f}, DICE {metrics1[6]:.4f}\n"
                    f"Seg2 : recall {100. * metrics2[0]:.4f}%, precision {100. * metrics2[1]:.4f}% "
                    f"acc {100. * metrics2[2]:.4f}%, fnr {100. * metrics2[3]:.4f}%, F1 {metrics2[4]:.4f},  "
                    f"IOU {metrics2[5]:.4f}, DICE {metrics2[6]:.4f}")
        ave_Loss_validate.append(loss)
        epoch += 1
        time.sleep(0.1)
    time_cost = (time.time() - st) / 60
    logger.info(f'-------------------------------\n'
                f'End of training, time cost : {time_cost // 60:.0f}h {time_cost % 60:.0f}m\n'
                f'-------------------------------')
    epoch = epoch - 1
    draw_comparision(ave_Loss, ave_Loss_validate, epoch, label1='training dataset', label2='validate dataset', index_x='epoch', index_y='loss', title='loss-epoch overall')
    plot_figure(lr_list, epoch, title='lr-epoch overall', x_label='epoch', y_label='lr')
    logger.info(f"ave_Loss : {ave_Loss}")
    logger.info(f"ave_Loss_validate : {ave_Loss_validate}")

    '''Test'''
    print('******test on test dataset******')
    net.eval()
    with open(F'TestResult\cost_result.csv', 'w', newline='') as csvfile:
        f = csv.writer(csvfile)
        f.writerow(["index", "prediction", "ground truth"])
    torch.cuda.empty_cache()
    Loss_test = 0
    metrics1 = [0, 0, 0, 0, 0, 0, 0]
    metrics2 = [0, 0, 0, 0, 0, 0, 0]
    metrics_all = [0, 0, 0, 0, 0, 0, 0]
    break_epoch = ave_Loss_validate.index(min(ave_Loss_validate)) + 1
    weights = save_weight_path % break_epoch
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        logger.info(f'-------------------------------------\n'
                    f"successfully load weights on break epoch: {weights}\n"
                    f"-------------------------------------")
    else:
        logger.info('unsuccessfully load weights')
    l = 0
    for (image, seg1, seg2, cost, coord) in tqdm(test_loader, desc='Testing', colour='red'):
        with torch.no_grad():
            image, seg1, seg2, cost = image.to(device), seg1.to(device), seg2.to(device), cost.to(device)
            # prom = net(image)
            prom, line, out_cost = net(image)
            '''
            for j in range(batch_num_test):
                predict_img = prom[j]
                true_img = seg1[j]
                input_map = image[j]
                unit_image = torch.stack([predict_img, true_img], dim=0)
                save_image(input_map, f'{test_image_path}/map_{j}.png')
                save_image(unit_image, f'{test_image_path}/{j}.png')
            '''
            train_loss = MSE_loss(torch.squeeze(out_cost, dim=1), cost)
        Loss_test += train_loss.item()
        recall, precision, acc, fnr, F1, IOU, DICE = evaluation(prom, seg1)
        metrics1 = [m + r.item() for m, r in zip(metrics1, [recall, precision, acc, fnr, F1, IOU, DICE])]
        recall, precision, acc, fnr, F1, IOU, DICE = evaluation(line, seg2)
        metrics2 = [m + r.item() for m, r in zip(metrics2, [recall, precision, acc, fnr, F1, IOU, DICE])]
        metrics_all = [x + y for x, y in zip(metrics1, metrics2)]
        for j in range(batch_num_test):
            img = torch.squeeze(image[j])
            predict_img = torch.squeeze(line[j])
            predict_img1 = torch.stack([predict_img, predict_img], dim=0)
            true_img = torch.squeeze(seg2[j])
            true_img1 = torch.stack([true_img, true_img], dim=0)
            save_image_tensor2pillow(true_img1, F'TestResult/Result1/{l}_{j}_line_gt.jpg')
            original_img = image[j]
            unit_image2 = torch.stack([original_img[0] - 1 / 255 * torch.squeeze(predict_img), original_img[1] - 127 / 255 * torch.squeeze(predict_img), original_img[2] - 127 / 255 * torch.squeeze(predict_img)], dim=0)
            unit_image2 = torch.squeeze(unit_image2)
            save_image_tensor2pillow(unit_image2, F'TestResult/Result1/{l}_{j}_line_v.jpg')
        for j in range(batch_num_test):
            img = torch.squeeze(image[j])
            predict_img = torch.squeeze(prom[j])
            predict_img1 = torch.stack([predict_img, predict_img], dim=0)
            true_img = torch.squeeze(seg1[j])
            true_img1 = torch.stack([true_img, true_img], dim=0)
            save_image_tensor2pillow(true_img1, F'TestResult/Result1/{l}_{j}_region_gt.jpg')
            original_img = image[j]
            unit_image2 = torch.stack([original_img[0] - 1 / 255 * torch.squeeze(predict_img), original_img[1] - 127 / 255 * torch.squeeze(predict_img), original_img[2] - 127 / 255 * torch.squeeze(predict_img)], dim=0)
            unit_image2 = torch.squeeze(unit_image2)
            save_image_tensor2pillow(unit_image2, F'TestResult/Result1/{l}_{j}_region_v.jpg')
            with open(F'TestResult/cost_result.csv', 'a', newline='') as csvfile:
                f = csv.writer(csvfile)
                f.writerow(["{}_{}".format(l, j), int(out_cost[j].view(-1).detach().cpu().numpy()),
                            int(cost[j].view(-1).detach().cpu().numpy())])
        l += 1
    loss = Loss_test / (len(test_loader))
    metrics1 = [m / (len(test_loader)) for m in metrics1]
    metrics2 = [m / (len(test_loader)) for m in metrics2]
    metrics_all = [m / (2 * len(test_loader)) for m in metrics_all]
    logger.info(f"[test results] regression loss:{loss:.4f}, "
                f"recall:{100. * metrics_all[0]:.4f}%, "
                f"precision:{100. * metrics_all[1]:.4f}%, "
                f"acc:{100. * metrics_all[2]:.4f}%, "
                f"fnr:{100. * metrics_all[3]:.4f}%, "
                f"F1:{metrics_all[4]:.4f}, "
                f"IOU:{metrics_all[5]:.4f}, "
                f"DICE:{metrics_all[6]:.4f}")
    logger.info(f"-----------------SEG1--------------------\n"
                f"[test results for SEG1] recall:{100. * metrics1[0]:.4f}%, "
                f"precision:{100. * metrics1[1]:.4f}%, "
                f"acc:{100. * metrics1[2]:.4f}%, "
                f"fnr:{100. * metrics1[3]:.4f}%, "
                f"F1:{metrics1[4]:.4f}, "
                f"IOU:{metrics1[5]:.4f}, "
                f"DICE:{metrics1[6]:.4f}")
    logger.info(f"-----------------SEG2--------------------\n"
                f"[test results for SEG2] recall:{100. * metrics2[0]:.4f}%, "
                f"precision:{100. * metrics2[1]:.4f}%, "
                f"acc:{100. * metrics2[2]:.4f}%, "
                f"fnr:{100. * metrics2[3]:.4f}%, "
                f"F1:{metrics2[4]:.4f}, "
                f"IOU:{metrics2[5]:.4f}, "
                f"DICE:{metrics2[6]:.4f}")