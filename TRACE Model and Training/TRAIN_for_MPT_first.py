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
batch_num = 2
batch_num_validate = batch_num
batch_num_test = batch_num
epoch = 1
epoch_num = 68
warmup_epoch = 6
model_args = dict(
    n_layers=4,
    n_heads=3,
    d_k=256,
    d_v=128,
    d_model=256,
    d_inner=512,
    pad_idx=None,
    n_position=16 * 16,
    dropout=0.2,
    train_shape=[16, 16],
)
net = Transformer1(**model_args).to(device)
data_path = 'Data_TSP'
weights_pre = r'best_result\mpt_.pth'
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
    logger = Logger(txt_path)
    if os.path.exists(weights_pre):
        try:
            net.load_state_dict(torch.load(weights_pre))
            logger.info(f'successfully load weights: {weights_pre}, epoch number = {epoch_num - 1}')
        except RuntimeError as e:
            print('Error:', e)
    else:
        print('No pretrained weights path')
    opt = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)
    scheduler = lr_scheduler.LinearLR(opt, start_factor=0.2, end_factor=1.0, total_iters=warmup_epoch)
    Focal_loss = FocalLoss(alpha=0.25, gamma=2).to(device)
    SoftDice_loss = SoftDiceLoss().to(device)
    MSE_loss = nn.MSELoss(reduction="mean").to(device)
    BCE_loss = nn.BCELoss().to(device)
    focal_loss2 = FocalLoss(alpha=2, gamma=0).to(device)

    ave_LISTs = lr_list, ave_Loss, ave_recall, ave_precision, ave_acc, ave_fnr, ave_F1, ave_IOU, ave_DICE, ave_Loss_validate, ave_recall_validate, ave_precision_validate, ave_acc_validate, ave_fnr_validate, ave_F1_validate, ave_IOU_validate, ave_DICE_validate = [
        [] for _ in range(17)]
    ave_NAMEs = ['learning rate', 'ave_Loss', 'ave_recall', 'ave_precision', 'ave_acc', 'ave_fnr', 'ave_F1', 'ave_IOU',
                 'ave_DICE', 'ave_Loss_validate', 'ave_recall_validate', 'ave_precision_validate', 'ave_acc_validate',
                 'ave_fnr_validate', 'ave_F1_validate', 'ave_IOU_validate', 'ave_DICE_validate']
    ave_evaluation_criteria_validate = [0, 0, 0, 0, 0, 0, 0]
    alpha = 2
    st = time.time()
    while epoch < epoch_num:

        '''Train'''
        since = time.time()
        torch.cuda.empty_cache()
        net.train()
        total_Loss = 0
        metrics = [0, 0, 0, 0, 0, 0, 0]
        for (image, seg1, seg2, cost, coord) in tqdm(data_loader, desc='Training', colour='blue'):
            image, seg1, seg2, cost = image.to(device), seg1.to(device), seg2.to(device), cost.to(device)
            prom, out_line, out_cost = net(image)
            train_loss = Focal_loss(prom, seg1) + SoftDice_loss(prom, seg1)
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            total_Loss += train_loss.item()
            recall, precision, acc, fnr, F1, IOU, DICE = evaluation(prom, seg1)
            metrics = [m + r.item() for m, r in zip(metrics, [recall, precision, acc, fnr, F1, IOU, DICE])]
        train_time = time.time() - since
        loss = total_Loss / (len(data_loader))
        metrics = [m / (len(data_loader)) for m in metrics]
        logger.info(
            f"Epoch {int(epoch)}/{int(epoch_num - 1)} , lr {scheduler.get_last_lr()[0]:.6f}------------------------------")
        logger.info(f"train: time {(train_time // 60):.0f}m {(train_time % 60):.0f}s, "
                    f"loss {loss:.4f}, recall {100. * metrics[0]:.4f}%, precision {100. * metrics[1]:.4f}% "
                    f"acc {100. * metrics[2]:.4f}%, fnr {100. * metrics[3]:.4f}%, F1 {metrics[4]:.4f},  "
                    f"IOU {metrics[5]:.4f}, DICE {metrics[6]:.4f}")
        if epoch == (warmup_epoch + 1):
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2, eta_min=1e-5)
        lr_list.append(scheduler.get_last_lr()[0])
        scheduler.step()
        ave_Loss.append(loss), ave_recall.append(metrics[0]), ave_precision.append(metrics[1]), ave_acc.append(
            metrics[2]), ave_fnr.append(metrics[3]), ave_F1.append(metrics[4]), ave_IOU.append(
            metrics[5]), ave_DICE.append(metrics[6])
        torch.save(net.state_dict(), save_weight_path % epoch)

        '''Eval'''
        torch.cuda.empty_cache()
        net.eval()
        val_start = time.time()
        Loss_validate = 0
        metrics = [0, 0, 0, 0, 0, 0, 0]
        total_evaluation_criteria_validate = [0, 0, 0, 0, 0, 0, 0]
        for (image, seg1, seg2, cost, coord) in tqdm(validate_loader, desc='Validating', colour='MAGENTA'):
            image, seg1, seg2, cost = image.to(device), seg1.to(device), seg2.to(device), cost.to(device)
            with torch.no_grad():
                prom, out_line, out_cost = net(image)
                train_loss = Focal_loss(prom, seg1) + SoftDice_loss(prom, seg1)
            Loss_validate += train_loss.item()
            recall, precision, acc, fnr, F1, IOU, DICE = evaluation(prom, seg1)
            metrics = [m + r.item() for m, r in zip(metrics, [recall, precision, acc, fnr, F1, IOU, DICE])]
        val_time = time.time() - val_start
        loss = Loss_validate / (len(validate_loader))
        metrics = [m / (len(validate_loader)) for m in metrics]
        logger.info(f"val: time {(val_time // 60):.0f}m {(val_time % 60):.0f}s, "
                    f"loss {loss:.4f}, "
                    f"recall {100. * metrics[0]:.4f}%, "
                    f"precision {100. * metrics[1]:.4f}% "
                    f"acc {100. * metrics[2]:.4f}%, "
                    f"fnr {100. * metrics[3]:.4f}%, "
                    f"F1 {metrics[4]:.4f},  "
                    f"IOU {metrics[5]:.4f}, "
                    f"DICE {metrics[6]:.4f}")
        ave_Loss_validate.append(loss), ave_recall_validate.append(metrics[0]), ave_precision_validate.append(
            metrics[1]), ave_acc_validate.append(metrics[2]), ave_fnr_validate.append(
            metrics[3]), ave_F1_validate.append(metrics[4]), ave_IOU_validate.append(
            metrics[5]), ave_DICE_validate.append(metrics[6])
        epoch += 1
        time.sleep(0.1)
    time_cost = (time.time() - st) / 60
    logger.info(f'-------------------------------\n'
                f'End of training, time cost : {time_cost // 60:.0f}h {time_cost % 60:.0f}m\n'
                f'-------------------------------')
    epoch = epoch - 1
    draw_comparision(ave_Loss, ave_Loss_validate, epoch, label1='training dataset', label2='validate dataset', index_x='epoch', index_y='loss', title='loss-epoch1')
    plot_figure(lr_list, epoch, title='lr-epoch1', x_label='epoch', y_label='lr')
    for name, lst in zip(ave_NAMEs, ave_LISTs):
        formatted = [float("{:.6f}".format(n)) for n in lst]
        logger.info(f"{name} : {formatted}")

    '''Test'''
    print('******test on test dataset******')
    net.eval()
    torch.cuda.empty_cache()
    Loss_test = 0
    metrics = [0, 0, 0, 0, 0, 0, 0]
    break_epoch = ave_F1_validate.index(max(ave_F1_validate)) + 1
    weights = save_weight_path % break_epoch
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        logger.info(f'-------------------------------------\n'
                    f"successfully load weights on break epoch: {break_epoch}\n"
                    f"-------------------------------------")
    else:
        logger.info('unsuccessfully load weights')
    i = 0
    for (image, seg1, seg2, cost, coord) in tqdm(test_loader, desc='Testing', colour='red'):
        with torch.no_grad():
            image, seg1, seg2, cost = image.to(device), seg1.to(device), seg2.to(device), cost.to(device)
            prom, out_line, out_cost = net(image)
            for j in range(batch_num_test):
                img = torch.squeeze(image[j])
                predict_img = torch.squeeze(prom[j])
                predict_img1 = torch.stack([predict_img, predict_img], dim=0)
                true_img = torch.squeeze(seg1[j])
                true_img1 = torch.stack([true_img, true_img], dim=0)
                save_image_tensor2pillow(true_img1, F'TestResult/Result1/1_groundtruth_{i}_{j}.jpg')
                original_img = image[j]
                unit_image1 = torch.stack([predict_img, true_img], dim=0)
                unit_image2 = torch.stack([original_img[0] - 1 / 255 * torch.squeeze(predict_img), original_img[1] - 127 / 255 * torch.squeeze(predict_img), original_img[2] - 127 / 255 * torch.squeeze(predict_img)], dim=0)
                unit_image2 = torch.squeeze(unit_image2)
                save_image_tensor2pillow(unit_image2, F'TestResult/Result1/1_original_{i}_{j}.jpg')
            i += 1
            train_loss = Focal_loss(prom, seg1) + SoftDice_loss(prom, seg1)
        Loss_test += train_loss.item()
        recall, precision, acc, fnr, F1, IOU, DICE = evaluation(prom, seg1)
        metrics = [m + r.item() for m, r in zip(metrics, [recall, precision, acc, fnr, F1, IOU, DICE])]
    loss = Loss_test / (len(test_loader))
    metrics = [m / (len(test_loader)) for m in metrics]
    logger.info(f"[test results] loss:{loss:.4f}, "
                f"recall:{100. * metrics[0]:.4f}%, "
                f"precision:{100. * metrics[1]:.4f}%, "
                f"acc:{100. * metrics[2]:.4f}%, "
                f"fnr:{100. * metrics[3]:.4f}%, "
                f"F1:{metrics[4]:.4f}, "
                f"IOU:{metrics[5]:.4f}, "
                f"DICE:{metrics[6]:.4f}")
