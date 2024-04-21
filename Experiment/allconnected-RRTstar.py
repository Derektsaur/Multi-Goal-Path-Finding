import csv

import PIL
import cv2
import random
import math
import time
import numpy as np
import copy
from PIL import Image
import elkai

class Node:
    #定义Node类
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None  #通过逆向回溯路径
        self.dis_to_root = 0
    def __repr__(self):
        return f'[{self.x}, {self.y}]'
    #  x是纵向方向， y是横向， 左上角是零点

#获得promsing region的坐标
# def get_promising(img_promising):
#     pro_list = []
#     for i in range(img_promising.shape[0]):
#         for j in range(img_promising.shape[1]):
#             if img_promising[i][j] > 200:     #因为promising region比较模糊
#                 pro_list.append(Node(i, j))
#
#     return pro_list

def random_sample_node(max_bound):
    return Node(random.randint(0, max_bound), random.randint(0, max_bound))

def get_dis(node1, node2): #两点之间的距离
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

def get_nearest_node(node_list, node):
    dis_list = [get_dis(nd, node) for nd in node_list]
    return node_list[dis_list.index(min(dis_list))]

def check_collision(img_binary, new_node):
    flag = False
    if img_binary[new_node.x][new_node.y] == 0:
        flag = True
    return flag

def check_path(img_binary, node1, node2):
    flag = False
    step_length = max(abs(node1.x - node2.x), abs(node1.y - node2.y))
    x = np.linspace(node1.x, node2.x, step_length + 1)
    y = np.linspace(node1.y, node2.y, step_length + 1)
    for i in range(step_length + 1):
        if check_collision(img_binary, Node(int(x[i]), int(y[i]))):
            flag = True
    return flag

def dis_to_root(node): #当前点到root距离
    total_dis = 0
    while True:
        if node.parent is not None:
            total_dis += get_dis(node, node.parent)
            node = node.parent
        else:
            break
    return total_dis

def nodes_in_range(target_node, range, node_list): #范围内的点,用于rrt*
    range_nodes = []
    for node in node_list:
        if get_dis(target_node, node) <= range and node is not target_node.parent:
            range_nodes.append(node)
    return range_nodes

def bias_rrtstar(target_first, target_second, bias_rate, img_binary, step_size,samplingMax):
    node_list = [target_first]
    num_sample_nodes = 0
    flag_path_found = False
    start_time = time.time()
    while True:

        num_sample_nodes += 1



        #sample在非region的point
        if random.random() > bias_rate:
            fg = True
            while fg:
                rand_node = random_sample_node(img_binary.shape[0] - 1)
                fg = False
            #rand_node = goal_node
        #sample在region的point
        else:
            rand_node = target_second

        #获取距离随机node最近的node
        nearest_node = get_nearest_node(node_list, rand_node)

        #生长角度与长度
        theta = math.atan2(rand_node.y - nearest_node.y, rand_node.x - nearest_node.x)
        expand_x = nearest_node.x + step_size * math.cos(theta)
        expand_y = nearest_node.y + step_size * math.sin(theta)

        #如果新node在图像外，去除
        if expand_x >= img_binary.shape[0] or expand_y >= img_binary.shape[1] or expand_x <= 0 or expand_y <= 0:
            continue
        #新node
        new_node = Node(int(expand_x), int(expand_y))
        new_node.parent = nearest_node
        #检查是否与障碍物重叠
        if check_collision(img_binary, new_node):
            continue
        #检查路径是否穿过障碍物
        if check_path(img_binary, new_node, new_node.parent):
            continue
        #此时，newnode生成了
        node_list.append(new_node)
        new_node.dis_to_root = nearest_node.dis_to_root + get_dis(new_node, nearest_node)
        #获取范围内的所有node
        # range_nodes = nodes_in_range(new_node, 60.0 * math.sqrt(50*(math.log(num_sample_nodes) / num_sample_nodes)), node_list)
        range_nodes = nodes_in_range(new_node, 5*step_size, node_list)
        #寻找newnode的新parent
        for nd in range_nodes:
            if get_dis(nd, new_node) + nd.dis_to_root < new_node.dis_to_root and nd is not target_second:
                if not check_path(img_binary, new_node, nd):
                    new_node.parent = nd
                    new_node.dis_to_root = nd.dis_to_root + get_dis(new_node, nd)

        #对范围内的节点检查，将newnode作为parent是否减少cost
        for nd in range_nodes:
            if nd is not new_node.parent:
                if get_dis(nd, new_node) + new_node.dis_to_root < nd.dis_to_root:
                    if not check_path(img_binary, new_node, nd):
                        nd.parent = new_node
                        nd.dis_to_root = new_node.dis_to_root + get_dis(nd, new_node)

        # 如果newnode与goalnode距离在expandDis以内
        if get_dis(new_node, target_second) < step_size * 3 and not flag_path_found:
            if check_path(img_binary, new_node, target_second):
                continue
            target_second.parent = new_node
            target_second.dis_to_root = new_node.dis_to_root + get_dis(target_second, new_node)
            flag_path_found = True
            cost = dis_to_root(target_second)

            # return time_consuming, cost, path, flag_path_found, node_list, num_sample_nodes
        if get_dis(new_node, target_second) < step_size * 3 and flag_path_found:
            if check_path(img_binary, new_node, target_second):
                continue
            if new_node.dis_to_root + get_dis(target_second, new_node)< target_second.dis_to_root:
                target_second.parent = new_node
                target_second.dis_to_root = new_node.dis_to_root + get_dis(target_second, new_node)
                cost = dis_to_root(target_second)

        if num_sample_nodes >= samplingMax and flag_path_found:
            node_list.append(target_second)
            node_path = target_second
            end_time = time.time()
            time_consuming = end_time - start_time
            path = []
            # 开始从goal回溯路径
            while True:
                if node_path.parent is not None:
                    path.append(node_path)
                    node_path = node_path.parent
                else:
                    path.append(target_first)
                    break
            return time_consuming, cost, path, flag_path_found, node_list, num_sample_nodes
        # if num_sample_nodes >= samplingMax and flag_path_found:
        #
        #     return time_consuming, cost, path, flag_path_found, node_list, num_sample_nodes


def draw_result(Img, start_node, goal_node, path):


    cv2.rectangle(Img, (start_node.y - 6, start_node.x - 6), (start_node.y + 6, start_node.x + 6), (255, 0, 0),
                  thickness=-1)
    cv2.rectangle(Img, (goal_node.y - 6, goal_node.x - 6), (goal_node.y + 6, goal_node.x + 6), (0, 0, 255),
                  thickness=-1)

    for nodes in path:
        if nodes.parent is not None:
            cv2.circle(Img, (nodes.y, nodes.x), 1, (0, 255, 0), -1)
            cv2.line(Img, (nodes.y, nodes.x), (nodes.parent.y, nodes.parent.x), (255, 0, 255), 1)


def draw_Tsp_result(Img, path):
    # print(path)
    cv2.rectangle(Img, (path[0].y - 6, path[0].x - 6), (path[0].y + 6, path[0].x + 6), (255, 0, 0),
                  thickness=-1)
    cv2.rectangle(Img, (path[len(path)-1].y - 6, path[len(path)-1].x - 6), (path[len(path)-1].y + 6, path[len(path)-1].x + 6), (255, 0, 0),
                  thickness=-1)
    for i in range(len(path)-1):
        # cv2.circle(Img, (nodes.y, nodes.x), 1, (0, 255, 0), -1)
        cv2.line(Img, (path[i].y, path[i].x), (path[i+1].y, path[i+1].x), (0, 0, 192), 2)

def draw_label(black_map, path):
    for i in range(len(path)-1):
        # cv2.circle(Img, (nodes.y, nodes.x), 1, (0, 255, 0), -1)
        cv2.line(black_map, (path[i].y, path[i].x), (path[i+1].y, path[i+1].x), (255, 255, 255), 30)


    # for nodes in path:
    # if nodes.parent is not None:
    # cv2.line(black_map, (nodes.y, nodes.x), (nodes.parent.y, nodes.parent.x), (255, 255, 255), 30)

def draw_maps_with_targets(IMG, target1_node, target2_node, target3_node, target4_node):
    cv2.rectangle(IMG, (target1_node.y - 5, target1_node.x - 5), (target1_node.y + 5, target1_node.x + 5), (255, 0, 0), thickness=-1)
    cv2.rectangle(IMG, (target2_node.y - 5, target2_node.x - 5), (target2_node.y + 5, target2_node.x + 5), (0, 0, 255), thickness=-1)
    cv2.rectangle(IMG, (target3_node.y - 5, target3_node.x - 5), (target3_node.y + 5, target3_node.x + 5), (0, 0, 255), thickness=-1)
    cv2.rectangle(IMG, (target4_node.y - 5, target4_node.x - 5), (target4_node.y + 5, target4_node.x + 5), (0, 0, 255), thickness=-1)


if __name__ == '__main__':

    """MAP"""
    # g = 2
    # targetCoordinate = [[18,37],[54,217],[226,28],[93,111],[247,125],[192,228]]
    g = 15
    targetCoordinate = [[226,40],[237,17],[81,171],[98,64],[175,222],[157,152],[88,29],[59,236],[141,220],[9,2],[199,216],[237,168],[68,11],[119,27],[11,105],[183,6],[187,175],[130,122],[13,162],[13,221],[207,108],[147,68],[238,207],[231,120],[37,249],[94,235],[158,137]]
    # g = 14
    # targetCoordinate = [[36,36],[50,168],[14,235],[131,165],[223,51],[196,235],[245,226],[98,94],[157,74],[235,124]]

    # g = 11
    # targetCoordinate = [[102, 18], [23, 23], [100, 120], [95, 177], [145, 145], [235, 34], [249, 130], [170, 230],
    #                     [14, 170], [26, 225], [240, 230], [200, 100], [81, 69], [191, 171]]


    # g = 36   #VVV   W1200
    # targetCoordinate = [[97,9],[56,86],[103,180],[21,25],[147,25],[96,84],[237,63],[14,224],[231,149],[245,219],[145,115],[93,245],[50,187],[185,185],[16,182],[210,246],[163,245],[215,15],[20,98],[47,140],[136,238]]

    # g = 31   # VVV    T3851
    # targetCoordinate = [[27,27],[21,236],[247,61],[174,25],[160,177],[231,175],[90,223],[69,135],[126,145]]
    with open(F'RESULT/RRT/RESULT_{g}_{len(targetCoordinate)}.csv','w',newline='') as csvfile:
        f = csv.writer(csvfile)
        f.writerow(["Test time","Calculation time","Path length","Sampling number",'Order'])
    for testNum in range(1):
        stat_time = time.time()
        map = cv2.imread("./MAP/%d.jpg" % g)
        white_map = cv2.imread("white.png")
        black_map = cv2.imread("black.png")
        black_map = cv2.resize(black_map, (256, 256))
        map_gray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
        ret, map_binary = cv2.threshold(map_gray, 127, 255, cv2.THRESH_BINARY)  # 二值图像0/255  201*201
        step_size = 10  # stepsize
        bias_rate = 0.1     #不选择region作为sample区域时
        samplingMax = 7000
        failureNum = 0
        totalSampleNum = 0

        city_number = len(targetCoordinate)
        target_group = []
        for i in range(city_number):
            target_node = Node(int(targetCoordinate[i][0]), int(targetCoordinate[i][1]))
            target_group.append(target_node)
        cost = np.zeros([city_number, city_number])
        path_final = list([[0 for i in range(city_number)] for j in range(city_number)])
        index =-1
        for i in range(len(target_group)):

            for j in range(len(target_group)):
                if i < j:
                    time_consuming_i_j, cost_i_j, path_i_j, flag_i_j,  node_list_i_j,num_sample_nodes= bias_rrtstar(target_group[i], target_group[j], bias_rate, map_binary, step_size,samplingMax)
                    print(num_sample_nodes)
                    totalSampleNum += num_sample_nodes
                    if not flag_i_j:
                        failureNum += 1
                        cost[i][j] = cost[j][i] = cost_i_j

                    else:
                        cost[i][j] = cost[j][i] = cost_i_j
                        path_final[i][j] = path_final[j][i] = path_i_j
                        for nodes in node_list_i_j:
                            nodes.parent = None

        # print(cost)
        M = cost
        list_final = []
        for i in range(len(elkai.solve_int_matrix(M))-1):
            temp = [elkai.solve_int_matrix(M)[i],elkai.solve_int_matrix(M)[i+1]]
            list_final.append(temp)
        list_final.append([elkai.solve_int_matrix(M)[len(elkai.solve_int_matrix(M)) - 1], elkai.solve_int_matrix(M)[0]])

        black_map_gray = cv2.cvtColor(black_map, cv2.COLOR_BGR2GRAY)
        IMG = copy.copy(map)
        # IMG2 = cv2.imread(r"C:\Users\huang\Desktop\S&Reg V2\PAPER\DRAW\BUILDING\W1200.png")     #T3851  W1200

        IMG_Black = copy.copy(black_map_gray)
        black_map_gray = cv2.cvtColor(black_map, cv2.COLOR_BGR2GRAY)
        black_map_gray = cv2.cvtColor(black_map, cv2.COLOR_BGR2GRAY)
        route = f'./RESULT/RRT/TSP_RESULT/{g}_{len(targetCoordinate)}_{testNum}.jpg'
        # route2 = f'./RESULT/RRT/TSP_result/{g}_{len(targetCoordinate)}_{testNum}_DRAW.jpg'

        cost_total = 0
        for k in range(len(list_final)):
            p = list_final[k][0]
            q = list_final[k][1]
            i = p
            j = q
            draw_Tsp_result(IMG, path_final[p][q])
            cv2.imwrite(route, IMG)
            # draw_Tsp_result(IMG2, path_final[p][q])
            # cv2.rectangle(IMG2, (targetCoordinate[0][1] - 6, targetCoordinate[0][0] - 6), (targetCoordinate[0][1] + 6, targetCoordinate[0][0] + 6), (0, 0, 255),
            #               thickness=-1)
            # cv2.imwrite(route2, IMG2)
            cost_total += cost[p][q]
        cv2.rectangle(IMG,(targetCoordinate[0][1] - 6, targetCoordinate[0][0] - 6), (targetCoordinate[0][1] + 6, targetCoordinate[0][0] + 6), (0, 0, 255), thickness=-1)
        cv2.imwrite(route, IMG)

        end_time = time.time()
        time_consuming = end_time - stat_time
        im = PIL.Image.fromarray(cv2.cvtColor(IMG,cv2.COLOR_BGR2RGB))
        im.save(route, quality=95,dpi=(300.0,300.0))
        # im = PIL.Image.fromarray(cv2.cvtColor(IMG2,cv2.COLOR_BGR2RGB))
        # im.save(route2, quality=95,dpi=(300.0,300.0))
        # print(g)
        # print(g)
        print("Calculation time: {}".format(time_consuming))
        print(cost_total)
        with open(F'RESULT/RRT/RESULT_{g}_{len(targetCoordinate)}.csv', 'a', newline='') as csvfile:
            f = csv.writer(csvfile)
            f.writerow([testNum, round(time_consuming,2), round(cost_total,2),
                        totalSampleNum,elkai.solve_int_matrix(M)])