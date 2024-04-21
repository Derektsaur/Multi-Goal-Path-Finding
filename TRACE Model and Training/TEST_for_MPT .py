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
epoch = 1  # 39
epoch_num = 68  # 65+39
warmup_epoch = 6
model_args = dict(
    n_layers=10,  # 6
    n_heads=6,  # 3
    d_k=256,  # 512
    d_v=128,  # 256
    d_model=256,  # 512
    d_inner=512,  # 1024
    pad_idx=None,
    n_position=16*16,  # 40*40
    dropout=0.1,
    train_shape=[16, 16],
)
net = Transformer(**model_args).to(device)
data_path = 'Data_TSP'  # \_Data_debug
save_weight_path = r'best_result/best_final_0815.pth' # r'best_result/best_thrid_0801.pth'
txt_path = 'figure_result'

if __name__ == '__main__':
    torch.cuda.empty_cache()
    cd = MyDataset(data_path)  # custom_dataset
    ts = int(len(cd) * 0.8)  # train_size 0.75:0.125:0.125
    vs = int(len(cd) * 0.1)  # validate_size
    test = int(len(cd)) - ts - vs  # test_size
    td, vd, ttd = torch.utils.data.random_split(cd, [ts, vs, test], generator=torch.Generator().manual_seed(0))
    data_loader = DataLoader(td, batch_size=batch_num, shuffle=True, pin_memory=True)
    validate_loader = DataLoader(vd, batch_size=batch_num_validate, shuffle=False, pin_memory=True)
    test_loader = DataLoader(ttd, batch_size=batch_num_test, shuffle=False, pin_memory=True)
    seed = 42
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger = Logger(txt_path)

    Focal_loss = FocalLoss(alpha=0.25, gamma=2).to(device)
    SoftDice_loss = SoftDiceLoss().to(device)
    MSE_loss = nn.MSELoss(reduction="mean").to(device)
    BEC_loss = nn.BCELoss().to(device)
    focal_loss2 = FocalLoss(alpha=2, gamma=0).to(device)

    # '''Validation'''
    # print('******test on val dataset******')
    # net.eval()
    # torch.cuda.empty_cache()
    # Reg_loss = 0
    # Prom_loss = 0
    # Line_loss = 0
    # metrics1 = [0, 0, 0, 0, 0, 0, 0]
    # metrics2 = [0, 0, 0, 0, 0, 0, 0]
    # if os.path.exists(save_weight_path):
    #     net.load_state_dict(torch.load(save_weight_path))
    #     logger.info(f'-------------------------------------\n'
    #                 f"successfully load weights for testing : {save_weight_path}\n"
    #                 f"-------------------------------------")
    # else:
    #     logger.info('unsuccessfully load weights')
    # i=0
    # for (image, seg1, seg2, cost, coord) in tqdm(validate_loader, desc='Testing', colour='red', ncols=100):
    #     with torch.no_grad():
    #         image, seg1, seg2, cost = image.to(device), seg1.to(device), seg2.to(device), cost.to(device)
    #         prom, line, out_cost = net(image)
    #         i+=1
    #
    #     reg_loss = MSE_loss(torch.squeeze(out_cost, dim=1), cost)
    #     prom_loss = Focal_loss(prom, seg1) + SoftDice_loss(prom, seg1)
    #     line_loss = Focal_loss(line, seg2)
    #
    #     Reg_loss += reg_loss.item()
    #     Prom_loss += prom_loss.item()
    #     Line_loss += line_loss.item()
    #     recall, precision, acc, fnr, F1, IOU, DICE = evaluation(prom, seg1)
    #     metrics1 = [m + r.item() for m, r in zip(metrics1, [recall, precision, acc, fnr, F1, IOU, DICE])]
    #     recall, precision, acc, fnr, F1, IOU, DICE = evaluation(line, seg2)
    #     metrics2 = [m + r.item() for m, r in zip(metrics2, [recall, precision, acc, fnr, F1, IOU, DICE])]
    #
    # reg_loss = Reg_loss / (len(validate_loader))
    # prom_loss = Prom_loss / (len(validate_loader))
    # line_loss = Line_loss / (len(validate_loader))
    # metrics1 = [m / (len(validate_loader)) for m in metrics1]
    # metrics2 = [m / (len(validate_loader)) for m in metrics2]
    # logger.info(f"[val results] reg_loss:{reg_loss:.4f}, prom_loss:{prom_loss:.4f}, line_loss:{line_loss:.4f}\n"
    #             f"Seg1 : recall {100. * metrics1[0]:.4f}%, precision {100. * metrics1[1]:.4f}% "
    #             f"acc {100. * metrics1[2]:.4f}%, fnr {100. * metrics1[3]:.4f}%, F1 {metrics1[4]:.4f},  "
    #             f"IOU {metrics1[5]:.4f}, DICE {metrics1[6]:.4f}\n"
    #             f"Seg2 : recall {100. * metrics2[0]:.4f}%, precision {100. * metrics2[1]:.4f}% "
    #             f"acc {100. * metrics2[2]:.4f}%, fnr {100. * metrics2[3]:.4f}%, F1 {metrics2[4]:.4f},  "
    #             f"IOU {metrics2[5]:.4f}, DICE {metrics2[6]:.4f}")

    '''Test'''
    print('******test on test dataset******')
    net.eval()
    torch.cuda.empty_cache()
    Reg_loss = 0
    Prom_loss = 0
    Line_loss = 0
    metrics1 = [0, 0, 0, 0, 0, 0, 0]
    metrics2 = [0, 0, 0, 0, 0, 0, 0]
    # logger = Logger(txt_path)
    if os.path.exists(save_weight_path):
        net.load_state_dict(torch.load(save_weight_path))
        logger.info(f'-------------------------------------\n'
                    f"successfully load weights for testing : {save_weight_path}\n"
                    f"-------------------------------------")
    else:
        logger.info('unsuccessfully load weights')
    i=0
    for (image, seg1, seg2, cost, coord) in tqdm(test_loader, desc='Testing', colour='red', ncols=100):
        with torch.no_grad():
            image, seg1, seg2, cost = image.to(device), seg1.to(device), seg2.to(device), cost.to(device)
            # prom = net(image)
            prom, line, out_cost = net(image)
            # '''
            # for j in range(batch_num_test):
            #     img = torch.squeeze(image[j])
            #     predict_img = torch.squeeze(prom[j])
            #     predict_img1 = torch.stack([predict_img, predict_img], dim=0)
            #     true_img = torch.squeeze(seg1[j])
            #     true_img1 = torch.stack([true_img, true_img], dim=0)
            #     save_image_tensor2pillow(predict_img1, F'TestResult/Result1/1_prediction{i}_{j}.jpg')
            #     save_image_tensor2pillow(true_img1, F'TestResult/Result1/1_groundtruth{i}_{j}.jpg')
            #
            #     original_img = image[j]
            #     unit_image1 = torch.stack([predict_img, true_img], dim=0)
            #     unit_image2 = torch.stack([original_img[0] - 1 / 255 * torch.squeeze(predict_img),
            #                                original_img[1] - 127 / 255 * torch.squeeze(predict_img),
            #                                original_img[2] - 127 / 255 * torch.squeeze(predict_img)], dim=0)
            #     unit_image2 = torch.squeeze(unit_image2)
            #     save_image_tensor2pillow(unit_image2, F'TestResult/Result1/1_{i}_{j}_original.jpg')
            #     save_image_tensor2pillow(unit_image1, F'TestResult/Result1/1_{i}_{j}.jpg')
            # predict_img = prom[j]
            # true_img = seg1[j]
            # input_map = image[j]
            # unit_image = torch.stack([predict_img, true_img], dim=0)
            # save_image(input_map, f'test_image_mine/map{i}_{j}.png')
            # save_image(unit_image, f'test_image_mine/{i}_{j}.png')
            # '''
            i+=1
            reg_loss = MSE_loss(torch.squeeze(out_cost, dim=1), cost)
            prom_loss = Focal_loss(prom, seg1) + SoftDice_loss(prom, seg1)
            line_loss = Focal_loss(line, seg2)

        Reg_loss += reg_loss.item()
        Prom_loss += prom_loss.item()
        Line_loss += line_loss.item()
        recall, precision, acc, fnr, F1, IOU, DICE = evaluation(prom, seg1)
        metrics1 = [m + r.item() for m, r in zip(metrics1, [recall, precision, acc, fnr, F1, IOU, DICE])]
        recall, precision, acc, fnr, F1, IOU, DICE = evaluation(line, seg2)
        metrics2 = [m + r.item() for m, r in zip(metrics2, [recall, precision, acc, fnr, F1, IOU, DICE])]

    reg_loss = Reg_loss / (len(test_loader))
    prom_loss = Prom_loss / (len(test_loader))
    line_loss = Line_loss / (len(test_loader))
    metrics1 = [m / (len(test_loader)) for m in metrics1]
    metrics2 = [m / (len(test_loader)) for m in metrics2]
    logger.info(f"[test results] reg_loss:{reg_loss:.4f}, prom_loss:{prom_loss:.4f}, line_loss:{line_loss:.4f}\n"
                f"Seg1 : recall {100. * metrics1[0]:.4f}%, precision {100. * metrics1[1]:.4f}% "
                f"acc {100. * metrics1[2]:.4f}%, fnr {100. * metrics1[3]:.4f}%, F1 {metrics1[4]:.4f},  "
                f"IOU {metrics1[5]:.4f}, DICE {metrics1[6]:.4f}\n"
                f"Seg2 : recall {100. * metrics2[0]:.4f}%, precision {100. * metrics2[1]:.4f}% "
                f"acc {100. * metrics2[2]:.4f}%, fnr {100. * metrics2[3]:.4f}%, F1 {metrics2[4]:.4f},  "
                f"IOU {metrics2[5]:.4f}, DICE {metrics2[6]:.4f}")
