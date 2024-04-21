
import time

from data import *
from torchvision.utils import save_image
import cv2
from torch.utils.data import DataLoader
from GPM import *
from torch.utils.data import Dataset

from utils import *
from torchvision import transforms

transform=transforms.Compose([transforms.ToTensor()])
######################
#Original map 201x201
#resize --> 256x256
#thus, cost-->cost*root(256/201)
######################



'''*************************************************'''
def keep_image_size_open_fortrain(path,size=(256,256)):
    img=Image.open(path)
    temp=max(img.size)
    # print(temp)
    mask=Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    mask=mask.resize(size)
    return mask

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
'''*************************************************'''



class PriorityNode:
    #定义Node类
    def __init__(self, x, y,start,goal,Flag=True):
        self.x = x
        self.y = y
        if Flag:
            self.come = abs(x-start.x)+abs(y-start.y)
            self.go = abs(x-goal.x)+abs(y-goal.y)
        else:
            self.come = start
            self.go = goal
    def __len__(self):
        return len(self.x)
    def __repr__(self):
        return f'[{self.x}, {self.y}, {self.come}, {self.go}]'

def PrioritySampleListGeneration(Pri_List):
    Pri_List_Sorted= sorted(Pri_List, key=lambda x: x.come)
    return Pri_List_Sorted

class Node:
    #定义Node类
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None  #通过逆向回溯路径
        self.cost = 0
    def __repr__(self):
        return f'[{self.x}, {self.y}]'

def ExtractionRegion(out_array_dilate,img):
    promising_region_list = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if out_array_dilate[i][j]>0.5 and img[i][j] != 0:
                promising_region_list.append(Node(i,j))
    return promising_region_list
def ExtractionLine(out_array_dilate,img,StartNode,GoalNode):
    promising_region_list = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if out_array_dilate[i][j]>0.5 and img[i][j] != 0:
                promising_region_list.append(PriorityNode(i,j,StartNode,GoalNode))
    return promising_region_list


Theta = (256/201)**1/2 #放大系数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

net = GPM_post().to(device)
net.eval()
weights = r'C:\Users\huang\Desktop\Guide-Line  RRT star\Result\GPM\TRAIN_POST\unet_20.pth'
save_path = r"C:\Users\huang\Desktop\Guide-Line  RRT star\PathPlanningResult\GRRT"
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully load weights')
else:
    print('unsuccessfully load weights')
MapFile = open(r"C:\Users\huang\Desktop\Guide-Line  RRT star\PathPlanningResult\MAP\number.txt", "r")
MapList = MapFile.read()
MapList = MapList.split()
MapFile.close()
MapCount = len(open(r"C:\Users\huang\Desktop\Guide-Line  RRT star\PathPlanningResult\MAP\number.txt",'r').readlines())
for i in range(MapCount):
    MapNumber = int(MapList[i])
    Map = cv2.imread(r'C:\Users\huang\Desktop\Guide-Line  RRT star\PathPlanningResult\MAP\Map\%d.jpg'% MapNumber)
    CoordinateFile = open(r"C:\Users\huang\Desktop\Guide-Line  RRT star\PathPlanningResult\MAP\Coordinate/" + str(MapNumber) + ".txt", "r")
    data = CoordinateFile.read()
    data = data.split()
    StartNode = Node(int((float(data[0])+0.5)*(256/201)-0.5), int((float(data[1])+0.5)*(256/201)-0.5))
    GoalNode = Node(int((float(data[2])+0.5)*(256/201)-0.5), int((float(data[3])+0.5)*(256/201)-0.5))
    CoordinateFile.close()
    # 读取threshold
    CostFile = open(r"C:\Users\huang\Desktop\Guide-Line  RRT star\PathPlanningResult\MAP\Length/" + str(MapNumber) + ".txt", "r")
    data = CostFile.read()
    data = data.split()
    CostOptimal = int(float(data[0])*Theta)
    CostFile.close()
    TestPath = 'C:/Users\huang\Desktop\Guide-Line  RRT star\PathPlanningResult\MAP\Example/' + str(MapNumber) + ".jpg"

    image = keep_image_size_open_fortrain(TestPath)
    Map = cv2.resize(cv2.imread(TestPath),(256,256))
    MapGray = cv2.cvtColor(cv2.imread(TestPath), cv2.COLOR_BGR2GRAY)
    ret, MapBinary = cv2.threshold(MapGray, 127, 255, cv2.THRESH_BINARY)
    MapBinary = cv2.resize(MapBinary,(256,256))

    image = torch.unsqueeze(transform(image),dim=0)
    data_loader = DataLoader(image, batch_size=1, shuffle=False)
    image = image.to(device)
    with torch.no_grad():
        out1,out2 = net(image)
        img = torch.squeeze(image[0])
        predict_img = torch.squeeze(out1[0])
        predict_img1 = torch.stack([predict_img, predict_img], dim=0)
        save_image_tensor2pillow(predict_img1, F'PlanningTestResult/{MapNumber}Region1.jpg')

        original_img = image[0]
        unit_image2 = torch.stack([original_img[0] - 110 / 255 * torch.squeeze(predict_img),
                                   original_img[1] - torch.squeeze(predict_img),
                                   original_img[2] - torch.squeeze(predict_img)], dim=0)
        unit_image2 = torch.squeeze(unit_image2)
        save_image_tensor2pillow(unit_image2, F'PlanningTestResult/{MapNumber}Region2.jpg')

        img = torch.squeeze(image[0])
        predict_img = torch.squeeze(out2[0])
        predict_img2 = torch.stack([predict_img, predict_img], dim=0)
        save_image_tensor2pillow(predict_img2, F'PlanningTestResult/{MapNumber}Line1.jpg')

        original_img = image[0]
        unit_image2 = torch.stack([original_img[0] - 110 / 255 * torch.squeeze(predict_img),
                                   original_img[1] - torch.squeeze(predict_img),
                                   original_img[2] - torch.squeeze(predict_img)], dim=0)
        unit_image2 = torch.squeeze(unit_image2)
        save_image_tensor2pillow(unit_image2, F'PlanningTestResult/{MapNumber}Line2.jpg')
    out1 = out1.squeeze()
    out2 = out2.squeeze()
    RegionArray = out1.detach().cpu().numpy()
    LineArray = out2.detach().cpu().numpy()
    RegionList = ExtractionRegion(RegionArray, MapBinary)
    LineList = ExtractionLine(LineArray, MapBinary,StartNode,GoalNode) #参数可调
    for i in range(len(LineList)):
        cv2.circle(Map,(LineList[i].y,LineList[i].x),1,(125,0,0))
    cv2.imshow('a', Map)
    cv2.rectangle(Map,(StartNode.y-2,StartNode.x-2),(StartNode.y+2,StartNode.x+2), (0, 255, 255),thickness=-1)
    cv2.rectangle(Map,(GoalNode.y-2,GoalNode.x-2),(GoalNode.y+2,GoalNode.x+2), (0, 255, 0),thickness=-1)

    PriLineList = PrioritySampleListGeneration(LineList)
    #######################################
    #PriLineList：以list形式储存PriorityNode#
    #######################################
    for i in range(10):
        cv2.circle(Map,(PriLineList[(i+1)*int(len(LineList)/10)].y,PriLineList[(i+1)*int(len(LineList)/10)].x),4,(0,0,255),thickness = -1)
        cv2.imshow('a', Map)
        cv2.waitKey()
