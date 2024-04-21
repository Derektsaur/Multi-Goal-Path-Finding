# 图像生成 与 记录parent（已修改储存结构：list）
# openList [tree1[Node1[x1,y1,selectNum,treeNum,parent,cost],Node2[x2,y2,...]],tree2,...]
# treeList [tree1[Node1[x1,y1,parent,cost],Node2[...]],tree2,...] parent = index of the connection node in the same tree; cost: return to the starting point
# connectionList [tree1[connection1[treeNum,index1,index2,cost],connection2[...]],tree2,...] index:self,other
import cv2
import elkai
import numpy as np
import math
import random
from PIL import Image
from numpy.ma import count
import copy
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt


# tsp问题
class Solution:
    def __init__(self, X, start_node):
        self.X = X  # 距离矩阵
        self.start_node = start_node  # 开始的节点
        self.array = [[0] * (2 ** (len(self.X) - 1)) for i in range(len(self.X))]  # 记录处于x节点，未经历M个节点时，矩阵储存x的下一步是M中哪个节点

    def transfer(self, sets):
        su = 0
        for s in sets:
            su = su + 2 ** (s - 1)  # 二进制转换
        return su

    # tsp总接口
    def tsp(self):
        s = self.start_node
        num = len(self.X)
        cities = list(range(num))  # 形成节点的集合
        # past_sets = [s] #已遍历节点集合
        cities.pop(cities.index(s))  # 构建未经历节点的集合
        node = s  # 初始节点
        return self.solve(node, cities)  # 求解函数

    def solve(self, node, future_sets):
        # 迭代终止条件，表示没有了未遍历节点，直接连接当前节点和起点即可
        if len(future_sets) == 0:
            return self.X[node][self.start_node]
        d = 99999
        # node如果经过future_sets中节点，最后回到原点的距离
        distance = []
        # 遍历未经历的节点
        for i in range(len(future_sets)):
            s_i = future_sets[i]
            copy = future_sets[:]
            copy.pop(i)  # 删除第i个节点，认为已经完成对其的访问
            distance.append(self.X[node][s_i] + self.solve(s_i, copy))
        # 动态规划递推方程，利用递归
        d = min(distance)
        # node需要连接的下一个节点
        next_one = future_sets[distance.index(d)]
        # 未遍历节点集合
        c = self.transfer(future_sets)
        # 回溯矩阵，（当前节点，未遍历节点集合）——>下一个节点
        self.array[node][c] = next_one
        return d


# 计算两点间的欧式距离
def distance(vector1, vector2):
    d = 0;
    for a, b in zip(vector1, vector2):
        d += (a - b) ** 2
    return d ** 0.5

def initialOpenList(openList,targetCoordinate):
    for i in range(len(targetCoordinate)):
        openList[i].append([targetCoordinate[i][0],targetCoordinate[i][1],0,i,0,0])
    return openList

def initialTree(treeList,targetCoordinate):
    for i in range(len(targetCoordinate)):
        treeList[i].append([targetCoordinate[i][0],targetCoordinate[i][1],0,0])
    return treeList

def dis(qNew, q):
    distance = round(math.sqrt((q[0] - qNew[0]) ** 2 + (q[1] - qNew[1]) ** 2),2)
    return distance

def connectionRecord(qNew, qq,treeNum,i,j,treeList,connectionList,costPartial):
    #i other node tree  j index of othre node in tree[i]
    toatlCost = round(costPartial + qq[3] + dis(qNew, qq),2)
    flag = False
    for t in range(len(connectionList[treeNum])-1,0,-1):
        if connectionList[treeNum][t][0] == i:
            if toatlCost < connectionList[treeNum][t][3]:
                connectionList[treeNum].append([i, len(treeList[qNew[3]]), j, toatlCost])
                connectionList[i].append([treeNum, j, len(treeList[qNew[3]]), toatlCost])
            flag = True
            break
        else:
            flag = False

    if not flag:
        connectionList[treeNum].append([i, len(treeList[qNew[3]]), j, toatlCost])
        connectionList[i].append([treeNum, j, len(treeList[qNew[3]]), toatlCost])
    return connectionList

def checkExpansion(qNew,q,treeList,tresholdTrees,connectionList,mapBinary):
    # 1. not in self-tree *(distanceNew)
    # 2. not in connection with other trees *(tresholdTrees)
    # 3.
    flag = True
    treeNum = q[3]
    distanceNew = dis(qNew, q)
    costPartial = distanceNew + q[5]

    for i in range(len(treeList[treeNum])):
        if dis(qNew,treeList[treeNum][i]) < distanceNew:
            flag = False
            return flag,connectionList,costPartial


    for i in range(len(treeList)):
        if i != treeNum:
            for j in range(len(treeList[i])):
                qq = treeList[i][j]
                if dis(qNew,qq) < tresholdTrees:
                    #update the cost of qNew
                    #q from oenList qq from treeList
                    if checkPath(qNew,qq,mapBinary):
                        connectionList = connectionRecord(qNew, qq,treeNum,i,j,treeList,connectionList,costPartial)
                        treeList[q[3]].append([qNew[0], qNew[1], treeList[q[3]].index([q[0], q[1], q[4], q[5]]), costPartial])
                        flag = False
                        return flag, connectionList, costPartial
                    else:
                        flag = False
                        return flag, connectionList, costPartial
                        # if checkConnection(q,qNew,tresholdTrees):
                #     flag = False
                #     return flag
        else:
            continue
    return flag, connectionList, costPartial

# 设置好参数步长R和连接阈值就用不到
# 两树尝试连接时使用
def checkConnection(q,qNew,tresholdTrees):
    flag = True
    step_length = max(abs(qNew[0] - q[0]), abs(qNew[1] - q[1]))
    x = np.linspace(q[0], qNew[0], step_length + 1)
    y = np.linspace(q[1], qNew[1], step_length + 1)
    for i in range(step_length + 1):
        if dis(q,[int(x[i]), int(y[i])])< tresholdTrees:
            flag = False
            return flag
    return flag
def checkPath(qNew,q,mapBinary):
    flag = False
    step_length = max(abs(qNew[0] - q[0]), abs(qNew[1] - q[1]))
    x = np.linspace(qNew[0], q[0], step_length + 1)
    y = np.linspace(qNew[1], q[1], step_length + 1)
    for i in range(step_length + 1):
        if checkNode(mapBinary, [int(x[i]), int(y[i])],maxH,maxW):
            flag = True
        else:
            flag = False
            break
    return flag
def checkNode(mapBinary, q,maxH,maxW):
    flag = True
    if q[0]>= maxH or q[0] < 0 or q[1]>= maxW or q[1] < 0:
        flag = False
        return flag
    if mapBinary[q[0]][q[1]] == 0:
        flag = False
        return flag

    return flag



def expansion(expansionDis, q, mapBinary, treeList, tresholdTrees,maxH,maxW,map,c,success,connectionList):
    theta = math.pi*2*random.random()
    dis = expansionDis*random.random()
    qNew_x = int(q[0]+dis*math.cos(theta))
    qNew_y = int(q[1]+dis*math.sin(theta))
    q[2] += 1
    if checkNode(mapBinary,[qNew_x,qNew_y],maxH,maxW):
        qNew = np.array([qNew_x,qNew_y,0,q[3]])
        if checkPath(qNew,q,mapBinary):
            flag, connectionList, costPartial = checkExpansion(qNew, q, treeList, tresholdTrees, connectionList, mapBinary)
            if flag:
                # print(q)
                cv2.line(map,(qNew_y,qNew_x),(q[1],q[0]),c[q[3]],thickness=1)
                for k in range(len(openList)):
                    for t in range(len(openList[k])):
                        openNode = openList[k][t]
                        cv2.circle(map, (openNode[1], openNode[0]), 2, (0,0,0), -1)
                cv2.imshow('a',map)
                cv2.waitKey(1)
                treeList[q[3]].append([qNew[0],qNew[1],treeList[q[3]].index([q[0],q[1],q[4],q[5]]),costPartial])
                openList[q[3]].append([qNew[0],qNew[1],0,q[3],treeList[q[3]].index([q[0],q[1],q[4],q[5]]),costPartial])
                success = True

    return treeList,openList,success,connectionList
def makePath(treeList,connectionList):
    pathRaw = []
    costListRaw = []
    connectionMatrix = [[0] * (len(treeList)) for _ in range(len(treeList))]
    for i in range(len(connectionList)):
        for j in range(len(connectionList[i])-1, 0, -1):
            if connectionList[i][j][0] != i:
                if connectionMatrix[i][connectionList[i][j][0]] == 0:
                    connectionMatrix[i][connectionList[i][j][0]] = 1
                    connectionMatrix[connectionList[i][j][0]][i] = 1
                    pathParticle, cost = pathExtract(treeList, connectionList, i, j)
                    pathRaw.append(pathParticle)
                    costListRaw.append(cost)
    print(connectionMatrix)
    return pathRaw, connectionMatrix, costListRaw

def pathExtract(treeList, connectionList, i, j):
    treeNum1 = i
    treeNum2 = connectionList[i][j][0]
    connectIndex1 = connectionList[i][j][1]
    connectIndex2 = connectionList[i][j][2]
    # connectCost = connectionList[i][j][3]
    cost = connectionList[i][j][3]
    path1 = []
    path2 = []
    currentNode1 = treeList[treeNum1][connectIndex1]
    while currentNode1[2] != 0:
        path1.insert(0, currentNode1)
        currentNode1 = treeList[treeNum1][currentNode1[2]]
    path1.insert(0, currentNode1)
    path1.insert(0,treeList[treeNum1][0])

    currentNode2 = treeList[treeNum2][connectIndex2]
    while currentNode2[2] != 0:
        path2.append(currentNode2)
        currentNode2 = treeList[treeNum2][currentNode2[2]]
    path2.append(currentNode2)
    path2.append(treeList[treeNum2][0])

    path1.extend(path2)
    pathParticle = path1

    return pathParticle, cost


def pathPruning(mapBinary, pathRaw, wPruned):
    # append [[1,2],3,4], extend [1,2,3,4], insert [[1,2],3,4]
    pathPruned = [[] for i in range(len(pathRaw))]
    for i in range(len(pathRaw)):
        if len(pathRaw[i]) <= wPruned:
            pathPruned[i].extend(pathRaw[i])
            continue
        pathPrunedPartial = pathRaw[i]
        j = len(pathRaw[i]) - wPruned - 1

        while j >= wPruned:
            if checkPath(pathRaw[i][j - wPruned], pathRaw[i][j + wPruned], mapBinary):
                for t in range(j + wPruned - 1,j - wPruned + 1,-1):
                    pathPrunedPartial.remove(pathRaw[i][t])
                j -= 2*wPruned
                flag = True
            else:
                j -= 1
                flag = False
        # possible to design an adaptive parameter wPruned, changing/reducing if the j>= wPruned
        if flag:
            idxCurrent = j + wPruned
        else:
            idxCurrent = j + 1
        if checkPath(pathRaw[i][idxCurrent], pathRaw[i][0], mapBinary):
            for t in range(idxCurrent - 1, 1, -1):
                pathPrunedPartial.remove(pathRaw[i][t])
        pathPruned[i].extend(pathPrunedPartial)
    costList = []
    for i in range(len(pathPruned)):
        cost = 0
        for j in range(len(pathPruned[i])-1):
            cost += dis(pathPruned[i][j],pathPruned[i][j+1])
        costList.append(round(cost, 2))
    return pathPruned, costList

def drawRawResult(map,pathRaw,saveSize):
    for i in range(len(pathRaw)):
        for j in range(len(pathRaw[i])-1):
            cv2.line(map,(pathRaw[i][j][1],pathRaw[i][j][0]),(pathRaw[i][j+1][1],pathRaw[i][j+1][0]),(3, 0, 142),thickness=2)
            cv2.circle(map, (pathRaw[i][j][1],pathRaw[i][j][0]), 3, (0,0,0), -1)

        # mapTempSave = Image.fromarray(cv2.cvtColor(map, cv2.COLOR_BGR2RGB))
        # mapTempSave = mapTempSave.resize((saveSize, saveSize), Image.ANTIALIAS)
        # mapTempSave.save(F"saveImage/mapTSPRawResult_{i}.jpg", dpi=(300, 300), bbox_inches='tight', quality=95)

    cv2.imshow('a',map)
    cv2.waitKey()

    mapSave = Image.fromarray(cv2.cvtColor(map, cv2.COLOR_BGR2RGB))
    mapSave = mapSave.resize((saveSize, saveSize), Image.ANTIALIAS)
    mapSave.save(F"./RESULT/SFF/GROWTH/mapTSPRawResult.jpg", dpi=(300, 300), bbox_inches='tight', quality=95)

def drawPrunedResult(newMap,pathPruned,targetCoordinate,saveSize):
    cv2.destroyAllWindows()
    for i in range(len(pathPruned)):
        for j in range(len(pathPruned[i])-1):
            cv2.line(newMap,(pathPruned[i][j][1], pathPruned[i][j][0]), (pathPruned[i][j+1][1], pathPruned[i][j+1][0]), (3, 0, 142), thickness=2)
            # cv2.circle(newMap, (pathPruned[i][j][1], pathPruned[i][j][0]), 3, (0, 0, 0), -1)
            cv2.imshow('b', newMap)
            cv2.waitKey(1)

        # mapTempSave = Image.fromarray(cv2.cvtColor(newMap, cv2.COLOR_BGR2RGB))
        # mapTempSave = mapTempSave.resize((saveSize,saveSize),Image.ANTIALIAS)
        # mapTempSave.save(F"./RESULT/SFF/GROWTH/mapTSPPrunedResult_{i}.jpg", dpi=(300, 300), bbox_inches='tight', quality=95)
    for i in range(len(targetCoordinate)):
        cv2.rectangle(newMap, (targetCoordinate[i][1] - 4, targetCoordinate[i][0] - 4),
                      (targetCoordinate[i][1] + 4, targetCoordinate[i][0] + 4), (0, 0, 255), thickness=-1)
    cv2.imshow('b', newMap)
    cv2.waitKey()
    mapSave = Image.fromarray(cv2.cvtColor(newMap, cv2.COLOR_BGR2RGB))
    mapSave = mapSave.resize((saveSize,saveSize),Image.ANTIALIAS)
    mapSave.save(F"./RESULT/SFF/GROWTH/mapTSPPruneResult.jpg", dpi=(300, 300), bbox_inches='tight', quality=95)
def drawPrunedResultwithNode(newMapwithNode, pathPruned, targetCoordinate,saveSize):
    for i in range(len(pathPruned)):
        for j in range(len(pathPruned[i])-1):
            cv2.line(newMapwithNode,(pathPruned[i][j][1], pathPruned[i][j][0]), (pathPruned[i][j+1][1], pathPruned[i][j+1][0]), (3, 0, 142), thickness=2)
            # cv2.circle(newMap, (pathPruned[i][j][1], pathPruned[i][j][0]), 3, (0, 0, 0), -1)
            cv2.imshow('b', newMapwithNode)
            cv2.waitKey(1)

        # mapTempSave = Image.fromarray(cv2.cvtColor(newMapwithNode, cv2.COLOR_BGR2RGB))
        # mapTempSave = mapTempSave.resize((saveSize,saveSize),Image.ANTIALIAS)
        # mapTempSave.save(F"saveImage/mapTSPPrunedResultwithNode_{i}.jpg", dpi=(300, 300), bbox_inches='tight', quality=95)
    for i in range(len(targetCoordinate)):
        cv2.rectangle(newMapwithNode, (targetCoordinate[i][1] - 4, targetCoordinate[i][0] - 4),
                      (targetCoordinate[i][1] + 4, targetCoordinate[i][0] + 4), (0, 0, 255), thickness=-1)
    cv2.imshow('b', newMapwithNode)
    cv2.waitKey()
    # mapSave = Image.fromarray(cv2.cvtColor(newMapwithNode, cv2.COLOR_BGR2RGB))
    # mapSave = mapSave.resize((saveSize,saveSize),Image.ANTIALIAS)
    # mapSave.save(F"saveImage/mapTSPPrunedResultwithNode.jpg", dpi=(300, 300), bbox_inches='tight', quality=95)

def drawFinalResult(newMapFinal, pathPruned, list_final, targetCoordinate, saveSize):

    for i in range(len(list_final)):
        indexX1 = targetCoordinate[list_final[i][0]][0]
        indexY1 = targetCoordinate[list_final[i][0]][1]
        indexX2 = targetCoordinate[list_final[i][1]][0]
        indexY2 = targetCoordinate[list_final[i][1]][1]
        for j in range(len(pathPruned)):
            if (pathPruned[j][0][0] == indexX1 and pathPruned[j][0][1] == indexY1 and pathPruned[j][-1][0] == indexX2 and pathPruned[j][-1][1] == indexY2) or (pathPruned[j][0][0] == indexX2 and pathPruned[j][0][1] == indexY2 and pathPruned[j][-1][0] == indexX1 and pathPruned[j][-1][1] == indexY1):
                for t in range(len(pathPruned[j]) - 1):
                    cv2.line(newMapFinal,(pathPruned[j][t][1], pathPruned[j][t][0]), (pathPruned[j][t+1][1], pathPruned[j][t+1][0]), (3, 0, 142), thickness=2)




    cv2.imshow('b', newMapFinal)
    cv2.waitKey(1)


    for i in range(len(targetCoordinate)):
        cv2.rectangle(newMapFinal, (targetCoordinate[i][1] - 4, targetCoordinate[i][0] - 4),
                      (targetCoordinate[i][1] + 4, targetCoordinate[i][0] + 4), (0, 0, 255), thickness=-1)
        cv2.putText(newMapFinal,"{}".format(i),(targetCoordinate[i][1] - 3, targetCoordinate[i][0] - 3), cv2.FONT_HERSHEY_COMPLEX, 1, (3,0,142), 2)

    cv2.imshow('b', newMapFinal)
    cv2.waitKey()
    mapTempSave = Image.fromarray(cv2.cvtColor(newMapFinal, cv2.COLOR_BGR2RGB))
    mapTempSave = mapTempSave.resize((saveSize,saveSize),Image.ANTIALIAS)
    mapTempSave.save(F"./RESULT/SFF/GROWTH/TSPresult.jpg", dpi=(300, 300), bbox_inches='tight', quality=95)
    # mapSave = Image.fromarray(cv2.cvtColor(newMapwithNode, cv2.COLOR_BGR2RGB))
    # mapSave = mapSave.resize((saveSize,saveSize),Image.ANTIALIAS)
    # mapSave.save(F"saveImage/mapTSPPrunedResultwithNode.jpg", dpi=(300, 300), bbox_inches='tight', quality=95)

if __name__ == '__main__':
    # c = ((0, 0, 255),(255, 0, 255),(0, 255, 255),(0, 255, 0),(255, 0, 0),(125, 125, 125),(255, 255, 100))
    c = ((0, 0, 255),(255, 0, 255),(0, 255, 255),(0, 255, 0),(255, 0, 0),(125, 125, 125),(255, 255, 100),(196,196,196),(128,128,128),(55,56,60),(105,105,105),(169,169,169),(153,136,119),(79,79,47),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128))
    # c = ((128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128),(128,128,128))
    # 11
    targetCoordinate = [[102, 18], [23, 23], [100, 120], [95, 177], [145, 145], [235, 34], [247, 130], [170, 230],
                        [14, 170], [26, 225], [240, 230], [200, 100]]
    # targetCoordinate = [[23, 23],  [95, 177], [145, 145], [235, 34], [247, 130], [170, 230],
    #                     [14, 170], [26, 225], [240, 230], [200, 100]]
    # targetCoordinate = [[23, 23],  [95, 177], [145, 145], [235, 34],  [170, 230],
    #                      [26, 225], [240, 230], [200, 100]]
    # targetCoordinate = [[23, 23],  [145, 145], [235, 34],
    #                      [26, 225], [240, 230], [200, 100]]
    # targetCoordinate = [[145, 145], [235, 34],
    #                      [26, 225], [240, 230]]

    # 9
    # targetCoordinate = [[170, 240], [90, 177],[12, 8], [236, 23],  [247, 130]]
    g = 11
    map = cv2.imread("./MAP/%d.jpg" % g)
    # map = cv2.imread(mapPath)
    map = cv2.resize(map,(256,256))
    mapGray = cv2.cvtColor(map,cv2.COLOR_BGR2GRAY)
    ers,mapBinary = list(cv2.threshold(mapGray,127,255,cv2.THRESH_BINARY))
    # coordinatePath = F""
    saveSize = 512
    newMap = copy.copy(map)
    newMapFinal = copy.copy(map)


    maxSelect = 8 #5:快 10：慢，但做图好看
    expansionDis = 13
    tresholdTrees = 25 #必须大于等于expansionDis
    sampleNum = 0
    wPruned = 5
    savePath = F"RESULT/SFF/GROWTH"
    maxH = np.array(mapBinary).shape[0]
    maxW = np.array(mapBinary).shape[1]

    for i in range(len(targetCoordinate)):
        cv2.rectangle(map,(targetCoordinate[i][1]-4,targetCoordinate[i][0]-4),(targetCoordinate[i][1]+4,targetCoordinate[i][0]+4),(0,0,255),thickness=-1)
    cv2.imshow('a',map)
    # cv2.waitKey()
    mapTSPSave = Image.fromarray(cv2.cvtColor(map, cv2.COLOR_BGR2RGB))
    mapTSPSave = mapTSPSave.resize((saveSize,saveSize),Image.ANTIALIAS)
    mapTSPSave.save(F"./RESULT/SFF/GROWTH/map.jpg",dpi=(300,300), bbox_inches='tight',quality=95)

    openList = [[] for i in range(len(targetCoordinate))]
    openList = initialOpenList(openList,targetCoordinate)
    treeList = [[] for i in range(len(targetCoordinate))]
    treeList = initialTree(treeList,targetCoordinate)
    connectionList = [[[i,0,0,0]] for i in range(len(targetCoordinate))]


    while count(openList)!=0:

        xIndex = random.randint(0,len(openList)-1)
        if len(openList[xIndex]) !=0:
            yIndex = random.randint(0,len(openList[xIndex])-1)
        else:
            sampleNum += 1
            continue
        q = openList[xIndex][yIndex]
        success = False
        while q[2] <= maxSelect and not success:
            treeList,openList,success,connectionList = expansion(expansionDis, q, mapBinary, treeList, tresholdTrees,maxH,maxW,map,c,success,connectionList)
        if q[2] > maxSelect:
            openList[xIndex].remove(q)
            cv2.circle(map, (q[1], q[0]), 2, c[q[3]], -1)

        sampleNum += 1
        # print(sampleNum)
        # print(treeList)
        # if sampleNum > 50000:
        #     print("forced quit")
        #     break
        # if count(np.array(openList,dtype=object))==0:
        #     break
    for i in range(len(treeList)):
        # for j in range(1,len(treeList[i])):
        #     q = treeList[i][j]
        #     cv2.circle(map,(q[1],q[0]),2,c,-1)
        cv2.rectangle(map, (targetCoordinate[i][1] - 4, targetCoordinate[i][0] - 4),
                      (targetCoordinate[i][1] + 4, targetCoordinate[i][0] + 4), (0,0,255), thickness=-1)
        cv2.putText(map,"{}".format(i),(targetCoordinate[i][1] - 5, targetCoordinate[i][0] - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (3,0,142), 2)
    cv2.imshow('a',map)
    cv2.waitKey()
    newMapwithNode = copy.copy(map)
    resultSave = Image.fromarray(cv2.cvtColor(map, cv2.COLOR_BGR2RGB))
    resultSave = resultSave.resize((saveSize, saveSize),Image.ANTIALIAS)
    resultSave.save(F"./RESULT/SFF/GROWTH/finalResult.jpg",dpi=(300,300), bbox_inches='tight',quality=95)
    pathRaw, connectionMatrix, costListRaw = makePath(treeList, connectionList)
    print(pathRaw)
    drawRawResult(map, pathRaw, saveSize)
    pathPruned, costList = pathPruning(mapBinary, pathRaw, wPruned)
    drawPrunedResult(newMap, pathPruned, targetCoordinate, saveSize)
    drawPrunedResultwithNode(newMapwithNode, pathPruned, targetCoordinate, saveSize)
    print(costListRaw)
    print(costList)
    costMatrix = [[10000] * (len(treeList)) for _ in range(len(treeList))]
    print(costMatrix)

    num = 0
    for r in range(len(connectionMatrix)):
        for c in range(len(connectionMatrix[r])):
            if c > r and connectionMatrix[r][c] == 1:
                costMatrix[r][c] = int(costList[num])
                costMatrix[c][r] = costMatrix[r][c]
                num += 1
            if r == c:
                costMatrix[r][c] = 0

    print(costMatrix)

    # print(elkai.solve_int_matrix(costMatrix))
    # list_final = []
    # for i in range(len(elkai.solve_int_matrix(costMatrix))-1):
    #     temp = [elkai.solve_int_matrix(costMatrix)[i],elkai.solve_int_matrix(costMatrix)[i+1]]
    #     list_final.append(temp)
    # list_final.append([elkai.solve_int_matrix(costMatrix)[len(elkai.solve_int_matrix(costMatrix)) - 1], elkai.solve_int_matrix(costMatrix)[0]])
    # print(list_final)
    # totalCost = 0
    # for [i,j] in list_final:
    #     print(costMatrix[i][j])
    #     totalCost += costMatrix[i][j]
    # print("Total length: {}".format(totalCost))
    # print(pathPruned)
    # print(list_final)

    S = Solution(costMatrix,0)
    print("最短距离：" + str(S.tsp()))
    # 开始回溯
    M = S.array
    lists = list(range(len(S.X)))
    start = S.start_node
    list_final = []
    while len(lists) > 0:
        lists.pop(lists.index(start))
        m = S.transfer(lists)
        next_node = S.array[start][m]
        print(start,"--->" ,next_node)
        list_final.append([start,next_node])
        start = next_node
    drawFinalResult(newMapFinal, pathPruned, list_final, targetCoordinate, saveSize)
