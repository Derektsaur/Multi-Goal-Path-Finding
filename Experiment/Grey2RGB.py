# 通过 cv2 函数
import cv2
import numpy as np

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#使用cv2读取图片
for i in range(21,39):
    num = i
    img = cv2.imread(r"C:\Users\huang\Desktop\S&Reg V2\CODE\MAP\{}.png".format(num),0)

    #将所有图片准换为同样大小
    img = 255*np.ones_like(img) - np.resize(img,(256,256))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j]<255:
                img[i][j]=0

    com = np.array([img, img, img])
    com=np.swapaxes(com,2,0)#转换三维矩阵X,Z轴的位置，将3放到Z轴上，这样才可以直接转换为RGB三色图
    com=np.swapaxes(com,0,1)#将X,Y轴调换位置，输出图片就是垂直方向的了
    com = Image.fromarray(com).convert('RGB')  # 将数组转化回图片s
    L = com.convert('1')

    L.save(r"C:\Users\huang\Desktop\S&Reg V2\CODE\MAP\{}.jpg".format(num), dpi=(300, 300), quality=95)  # 将数组保存为图片
