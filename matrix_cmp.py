import os,sys
import numpy as np
from scipy.sparse import coo_matrix
from collections import deque,defaultdict
# sys.setrecursionlimit(3000)
# np.set_printoptions(threshold=np.nan)
class TreeNode:
    def __init__(self, val, children=None):
        self.val = val
        self.children = children if children else []

# def adjacency_matrix_to_tree(matrix):
#     num_nodes = len(matrix)
#     # print(num_nodes)
#
#     root_index = find_root(matrix, num_nodes)
#     print(root_index)
#     # print(e)
#     return build_tree(matrix, root_index)

def find_root(matrix, num_nodes):
    # 找到只有入度而没有出度的节点，即根节点
    in_degrees = [0] * num_nodes

    for row in matrix:
        for col_index, value in enumerate(row):
            in_degrees[col_index] += value
    # print(in_degrees)
    # print(e)
    # print(in_degrees.index(0))
    return in_degrees.index(0)

def build_tree(matrix, node_index):
    # print(matrix)
    # print(node_index)
    # print(e)
    #取根节点的那行，如果对应那行出现1，代表有一个对应的连接
    children_indices = [i for i, value in enumerate(matrix[node_index]) if value == 1]
    # print(children_indices)
    # print(e)
    #进行广度遍历
    children = [build_tree(matrix, child_index) for child_index in children_indices]
    return TreeNode(node_index, children)

# # 示例邻接矩阵
# adjacency_matrix = [
#     [0, 1, 0, 0, 1, 1],
#     [0, 0, 1, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
# ]

# 转换为结构树
# root_node = adjacency_matrix_to_tree(adjacency_matrix)
#
# # 输出结构树的结构
# def tree_structure(node, depth=0):
#     if node:
#         print(" " * depth + str(node.val))
#         for child in node.children:
#             tree_structure(child, depth + 1)
#
# print("Structure Tree:")
# tree_structure(root_node)

# 构建结构树
#        A
#     /  |  \
#    B   C   D
#   / \
#  E   F
# tree1 = TreeNode([TreeNode([TreeNode(), TreeNode()]), TreeNode(), TreeNode()])

# 构建目标结构
#    B
#   / \
#  E   F
# adjacency_matrix_1 = [
#     [0, 1, 0, 0, 1, 1],
#     [0, 0, 1, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
# ]
#
# adjacency_matrix_2 = [
#     [0, 0, 0, 0],
#     [1, 0, 1, 0],#根
#     [0, 0, 0, 0],
#     [0, 0, 0, 1],
#
# ]
#val里边存放的是下标


class Stack:
    def __init__(self):
        self.item = []

    def isEmpty(self):
        return self.item == []

    def push(self, element):
        self.item.append(element)

    def peek(self):
        return self.item[-1]

    def size(self):
        return len(self.item)

    def pop(self):
        top_element = self.item[-1]
        del self.item[-1]
        return top_element

global stack_S
stack_S = Stack()

#这里边要设置一个堆栈
def tree_cmp_1(tree_1,tree_2):
    import networkx as nx
    import matplotlib.pyplot as plt

    def has_structure(root1, root2):
        stack_S.push([root1.val,root2.val])
        if not root2:
            return True
        if not root1:
            stack_S.pop()
            return False

        if len(root1.children) < len(root2.children):
            stack_S.pop()
            return False

        return all(has_structure(child1, child2) for child1, child2 in zip(root1.children, root2.children))

    # 构建结构树
    #        A
    #     /  |  \
    #    B   C   D
    #   / \
    #  E   F
    # tree1 = TreeNode([TreeNode([TreeNode(), TreeNode()]), TreeNode(), TreeNode()])

    # 构建目标结构
    #    B
    #   / \
    #  E   F
    # structure = TreeNode([TreeNode(), TreeNode(), TreeNode(), ])
    is_structure = has_structure(tree_1, tree_2)

    if is_structure:
        print("tree1 contains the target structure")

    else:
        print("tree1 does not contain the target structure")
    return is_structure

def adjacency_matrix_to_tree1(adjacency_matrix):
    num_vertices = len(adjacency_matrix)
    visited = [False] * num_vertices
    root = None
    root_index = find_root(adjacency_matrix, num_vertices)
    # print(root_index)

    queue = deque()
    #这里默认第一个节点为根节点
    #实际上并不是
    #逐渐取每个点
    node_set = {}
    for i in range(num_vertices):
        node_set[i] = i
    node_set[0] = root_index
    node_set[root_index] = 0

    for vertex in range(len(node_set)):
        if not visited[node_set[vertex]]:
            root = TreeNode(node_set[vertex])
            queue.append(root)

            while queue:
                current_node = queue.popleft()
                current_vertex = current_node.val
                visited[current_vertex] = True

                for neighbor in range(num_vertices):
                    if adjacency_matrix[current_vertex][neighbor] == 1 and not visited[neighbor]:
                        child_node = TreeNode(neighbor)
                        current_node.children.append(child_node)
                        queue.append(child_node)

    return root

def sparse_to_dense(matrix_1,matrix_2,num):
    # row = np.array([0, 3, 1, 0])
    # col = np.array([0, 3, 1, 2])
    # print(matrix_2)
    # print(matrix_1)
    row = matrix_1
    col = matrix_2
    # print(row)
    # print(col)
    # data = np.array([4, 5, 7, 9])
    # num = int(max(col))
    max = 0
    for col_num in col:
        col_num = int(col_num)
        if col_num>max:
            max = col_num
    num = max


    row_num = row.shape[0]
    # print(row_num)
    data = np.ones((1,row_num),dtype=int)[0]
    # print(len(row))
    # print(len(col))
    # print(row_num)
    # print(num)
    # print(row_num)
    matrix = coo_matrix((data, (row, col)), shape=(num+1, num+1)).toarray()
    # print(len(matrix[0]))
    # print(matrix.shape)
    return matrix

def depth_adjacency_matrix(graph, start_vertex):
    id_dep_map={}
    num_vertices = len(graph)
    visited = [False] * num_vertices
    stack = [(start_vertex, 0, None)]  # Each entry: (vertex, depth)

    while stack:
        current_vertex, depth, parent = stack.pop()
        id_dep_map[current_vertex] = [depth,parent]
        # print(f"Vertex {current_vertex} - Depth: {depth}")

        visited[current_vertex] = True

        for neighbor in range(num_vertices):
            if graph[current_vertex][neighbor] == 1 and not visited[neighbor]:
                stack.append((neighbor, depth + 1,current_vertex))
    return id_dep_map
    # 示例邻接矩阵


    # start_vertex = 0
    # print("Depth of each vertex starting from vertex", start_vertex)
    # depth_adjacency_matrix(adjacency_matrix, start_vertex)


def depth_adjacency_matrix_bigcn(graph, start_vertex):
    id_dep_map={}
    num_vertices = len(graph)
    visited = [False] * num_vertices
    count = 1
    stack = [(start_vertex, 0, None)]  # Each entry: (vertex, depth)
    # print('stae',start_vertex)


    while stack:
        current_vertex, depth, parent = stack.pop()
        id_dep_map[current_vertex] = [depth,parent]
        # print(f"Vertex {current_vertex} - Depth: {depth}")

        visited[current_vertex] = True

        for neighbor in range(num_vertices):
            if graph[current_vertex][neighbor] == 1 and not visited[neighbor]:
                count+=1
                stack.append((neighbor, depth + 1,current_vertex))
    if count!=num_vertices:
        print(e)
    return id_dep_map
    # 示例邻接矩阵


    # start_vertex = 0
    # print("Depth of each vertex starting from vertex", start_vertex)
    # depth_adjacency_matrix(adjacency_matrix, start_vertex)


def get_matrix(dataset_name):
    # 首先先取出两个文件中对应的数组
    Project_path = "E:/pyProjects/BiGCN-master/"

    rvnn_path = Project_path + "rumor_detection_acl2017/"
    bigcn_path = Project_path + "Process/data/"

    # print("Loading {}".format(path))
    path_rvnn = rvnn_path+dataset_name+"matrix/"
    path_bigcn = bigcn_path+dataset_name+"matrix/"

    if path_rvnn[-1] == '/':
        #把
        rvnn_files = sorted([path_rvnn + f for f in os.listdir(path_rvnn)],
                            key=lambda x: int(x.split('/')[-1].split('.')[0]))  # 用idx.pkl中的idx排序
        bigcn_files = sorted([path_bigcn + f for f in os.listdir(path_bigcn)],
                            key=lambda x: int(x.split('/')[-1].split('.')[0]))  # 用idx.pkl中的idx排序
        # files = files[: DEBUG_NUM] if DEBUG else files
        # rvnn_files = [file for file in tqdm(files)]

    # print("Preprocessing {}".format(path))

    # for file in files:
    # print(len(rvnn_files))
    # print(len(bigcn_files))
    count = 0
    error_file = []
    for i in range(len(rvnn_files)):
        # try:

        rvnn_file = rvnn_files[i]
        bigcn_file = bigcn_files[i]
        id = int(rvnn_file.split('/')[-1].split('.')[0])

        rvnn_file_data = np.load(os.path.join(rvnn_file), allow_pickle=True)
        bigcn_file_data = np.load(os.path.join(bigcn_file), allow_pickle=True)
        # np.savez(os.path.join(cwd, 'data/' + obj + 'matrix/' + id + '.txt'), num=rootfeat.shape[0], edgeindex=tree,
        #          rootindex=rootindex)
        # np.savez(os.path.join(rvnn_path,  dataset_name + 'matrix/' + str(id) + '.txt'), num=idx, edgeindex=tree,
        #          rootindex=rootindex)
        #TODO rvnn里边还包括了时间数组
        rvnn_num, rvnn_matrix, rvnn_root = rvnn_file_data['num'],rvnn_file_data['edgeindex'],rvnn_file_data['rootindex']
        bigcn_num, bigcn_matrix, bigcn_root = bigcn_file_data['num'], bigcn_file_data['edgeindex'], bigcn_file_data['rootindex']
        #接下来先尝试把矩阵构建出来
        # print(rvnn_num)
        # print(rvnn_matrix)
        # print(bigcn_num)
        # print(bigcn_matrix)
        print('---------------------------------')

        print(rvnn_file)
        # print(bigcn_file)
        # if '407173794583695360' not in rvnn_file:
        #     continue
        # print(bigcn_matrix)
        if len(bigcn_matrix[0]) == 0:
            new_bigcn_time = []
        # print(len(bigcn_matrix))
        # print(bigcn_matrix)
        else:
            dense_rvnn_matrix = sparse_to_dense(rvnn_matrix[0],rvnn_matrix[1],rvnn_num)
            dense_bigcn_matrix = sparse_to_dense(bigcn_matrix[0],bigcn_matrix[1],bigcn_num)
            # print(dense_rvnn_matrix.shape)
            # print(dense_bigcn_matrix.shape)
            # print(e)

            # root_node_1 = adjacency_matrix_to_tree1(dense_rvnn_matrix)
            # root_node_2 = adjacency_matrix_to_tree1(dense_bigcn_matrix)
            # if len(dense_bigcn_matrix)>5:
            #     continue
            # else:
            # f = open(rvnn_path+'/1.txt','w')
            # # f.write(str(dense_rvnn_matrix))
            # for i in dense_rvnn_matrix:
            #     f.write('[')
            #
            #     for j in i:
            #         f.write(str(j)+",")
            #     f.write('],\n')
            # f.close()
            # print(dense_rvnn_matrix)
            # print(len(dense_rvnn_matrix))

            # print(dense_bigcn_matrix)
            # is_struct = tree_cmp_1(root_node_1,root_node_2)
            num_vertices = len(dense_rvnn_matrix)
            # print('bigcn',len(dense_bigcn_matrix))
            # visited = [False] * num_vertices
            # root = None
            rvnn_root_index = find_root(dense_rvnn_matrix,num_vertices)
            rvnn_id_dep_map = depth_adjacency_matrix(dense_rvnn_matrix,rvnn_root_index)

            new_rvnn_time = np.insert(rvnn_matrix[2],0,'0.0')

            #先计算一下总共有几个节点
            # print(len(new_rvnn_time))
            # print(len(dense_rvnn_matrix))
            if len(new_rvnn_time)<len(dense_rvnn_matrix):
                min = len(dense_rvnn_matrix) - len(new_rvnn_time)
                for i in range(min):
                    new_rvnn_time = np.insert(new_rvnn_time,-1,new_rvnn_time[-1])
            # print(len(new_rvnn_time))
            # print(len(dense_rvnn_matrix))
            rvnn_new_map=defaultdict(list)
            # print(len(new_rvnn_time))
            # print(len(rvnn_matrix))


            #
            for key,val in rvnn_id_dep_map.items():
                # if val not in new_map:
                # print()
                # print('key,val',key,val)
                # print('eeeeeee:',val)
                # print(val[1])
                rvnn_new_map[val[0]].append([key,new_rvnn_time[key],val[1]])
            # print(new_map)

            for key ,val in rvnn_new_map.items():
                # print(rvnn_new_map[key])
                rvnn_new_map[key].sort(key=lambda x: float(x[1]))
                            # new_map
            # print(rvnn_new_map)

            bigcn_new_map = defaultdict(list)
            num_vertices = len(dense_bigcn_matrix)

            bigcn_root_index = find_root(dense_bigcn_matrix, num_vertices)

            bigcn_id_dep_map = depth_adjacency_matrix_bigcn(dense_bigcn_matrix, bigcn_root_index)

            # print(bigcn_id_dep_map)
            for key,val in bigcn_id_dep_map.items():
                # if val not in new_map:
                # print()
                bigcn_new_map[val[0]].append([key,val[1]])
            # print(bigcn_new_map)

            select_rvnn_node = []
            #遍历两个字典里边度相同的节点
            # print(bigcn_new_map)
            bigcn_time_map = defaultdict(list)
            for key,val in bigcn_new_map.items():
                for i in bigcn_new_map[key]:
                    # print(i)
                    # time = rvnn_new_map[key].index()
                    #查找对应的时间
                    #rvnn 下标 时间 父节点
                    bigcn_node, bigcn_parent = i[0],i[1]
                    # print(i)
                    for j in rvnn_new_map[key]:
                        # print(j)
                        rvnn_node,rvnn_time,rvnn_parent = j[0],j[1],j[2]
                        if rvnn_node  in select_rvnn_node:
                            continue
                        # if
                        if bigcn_parent == None:
                            bigcn_time_map[key].append([bigcn_node,rvnn_time,bigcn_parent])
                        # print(e)
                        else:
                            #否者先判断父亲的时间
                            # print(bigcn_time_map)
                            # print(key - 1)
                            # print(bigcn_time_map[key - 1])
                            # print(bigcn_parent)
                            # print(bigcn_time_map[key-1])
                            for k in bigcn_time_map[key - 1]:
                                if k[0] == bigcn_parent:
                                    bigcn_time_map_node = k[0]
                                    bigcn_time_map_time = k[1]
                                    break

                            #如果当前rvnn节点的值小于父节点的值则不做
                            if rvnn_time <= bigcn_time_map_time:
                                # break
                                continue
                            #如果rvnn大于父节点时间
                            bigcn_time_map[key].append([bigcn_node,rvnn_time,bigcn_parent])
                            #然后要把节点从rvnn中剔除


                        select_rvnn_node.append(rvnn_node)
                        break

                error_file.append(rvnn_file)
                # count+=1
                # continue
                        # if float(j[1])
            #现在拥有了每个节点对应的时间，接下来去除原有的bigcn中的节点信息，给每个节点重新赋值上时间信息，存入到一个新的文件夹中，
            #然后在取出时间文件夹中的数据按照时间节点进行划分
            # print(bigcn_time_map)
            #现在有了数据要往回写数据
            #        new_rvnn_time = np.insert(rvnn_matrix[2],0,'0.0')
            new_bigcn_time = []

            for i in range(len(bigcn_matrix[0])):
                #依次寻找每个时间点
                # print(i)
                bigcn_time_parent = bigcn_matrix[0][i]
                bigcn_time_child = bigcn_matrix[1][i]
                flag = 0
                for map1_key,map1_val in bigcn_time_map.items():
                    if flag:
                        break
                    for map2 in map1_val:
                        if flag:
                            break
                        map2_parent = map2[2]
                        map2_child = map2[0]
                        map2_time = map2[1]
                        if (bigcn_time_parent == map2_parent) and (bigcn_time_child == map2_child):
                            new_bigcn_time.append(map2_time)
                            flag = 1

        # print(new_bigcn_time)
        # new_bigcn_file_data = np.load(os.path.join(bigcn_file), allow_pickle=True)
        # tree, rootindex=  np.array(tree), np.array(rootindex)
        #bigcn_num, bigcn_matrix, bigcn_root = bigcn_file_data['num'], bigcn_file_data['edgeindex'], bigcn_file_data['rootindex']
        new_bigcn_time = np.array(new_bigcn_time)
        np.savez(os.path.join(bigcn_path,  dataset_name + 'fulltime/' + str(id) + '.txt'),  edgeindex=bigcn_matrix, edgetime = new_bigcn_time,
                 rootindex=bigcn_root)

        #开始写入数据，首先去除对应的npz文件，然后往里边多添加一个数组

        # print(e)
    return count


print(get_matrix(dataset_name='Twitter15'))

