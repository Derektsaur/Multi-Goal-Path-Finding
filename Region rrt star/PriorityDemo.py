class PriorityNode:
    #定义Node类
    def __init__(self, x, y,start,goal):
        self.x = x
        self.y = y
        self.come = abs(x-start.x)+abs(y-start.y)
        self.go = abs(x-goal.x)+abs(y-goal.y)

    def __repr__(self):
        return f'[{self.x}, {self.y}, {self.come}, {self.go}]'

class Node:
    #定义Node类
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None  #通过逆向回溯路径
        self.dis_to_root = 0
    def __repr__(self):
        return f'[{self.x}, {self.y}]'

def PrioritySampleListGeneration(pro_list,start,goal):
    Pri_List = []
    for i in range(len(pro_list)):
        Pri_List.append(PriorityNode(pro_list[i].x,pro_list[i].y,start,goal))
    print(Pri_List)
    Pri_List_Sorted = sorted(Pri_List, key=lambda x: x.come)
    print(Pri_List_Sorted)




if __name__ == '__main__':
    pro_list = []
    for i in range(5):
        pro_list.append(Node(3*i,2*i+4))
    print(pro_list)
    start = Node(5,2)
    goal = Node(1,2)
    PrioritySampleListGeneration(pro_list,start,goal)