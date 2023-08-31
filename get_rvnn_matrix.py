import os
import tqdm
import numpy as np
import sys
cwd=os.getcwd()

project_path = "E:/pyProjects/BiGCN-master/"

rvnn_path = project_path+"rumor_detection_acl2017/"

def get_rvnn_matrix(dataset_name):
    # print("Loading {}".format(path))
    path = rvnn_path+dataset_name+"/tree/"

    if path[-1] == '/':
        #把
        files = sorted([path + f for f in os.listdir(path)],
                            key=lambda x: int(x.split('/')[-1].split('.')[0]))  # 用idx.pkl中的idx排序
        # files = files[: DEBUG_NUM] if DEBUG else files
        # rvnn_files = [file for file in tqdm(files)]

    print("Preprocessing {}".format(path))

    for file in files:

        uid2id_map = {}
        idx = 0
        # op_f = open(file,'r')
        row = []
        col = []
        time = []
        # if u_time == "0.0":

        id = int(file.split('/')[-1].split('.')[0])
        # print(id)
        # if '523123779124600833' not in file:
        #     continue
        print(file)
        line_id = 0
        with open(file,'r') as op_f:
            for line in op_f:

                uid = eval(line.strip('\n').split('->')[1])[0]
                parent_post_id = eval(line.strip('\n').split('->')[0])[1]

                child_post_id = eval(line.strip('\n').split('->')[1])[1]
                u_time = eval(line.strip('\n').split('->')[1])[2]
                if line_id == 0:
                    # print(uid)
                    rootindex = uid
                    twitter_id = child_post_id

                if (uid not in uid2id_map) and (child_post_id == twitter_id) and ((parent_post_id == twitter_id) or line_id==0):
                    uid2id_map[uid] = [idx,u_time]
                    idx+=1
                else:
                    pass
                line_id += 1
                    # print('error------------')
                    # print(id)
                    # print(uid)
        # op_f.close()
        with open(file, 'r') as op_f:
                # for line in op_f:
            for line in op_f:
                # print(line)
                line_lst = line.strip('\n').split('->')
                parent = eval(line_lst[0])
                child = eval(line_lst[1])
                # print(child)

                # print(uid2id_map[child[0]])
                if ('ROOT' in parent):
                    continue
                child_post_id = child[1]
                parent_post_id = parent[1]

                # print(twitter_id,parent_post_id,child_post_id)
                # if (child_post_id != twitter_id) or (parent_post_id != twitter_id) or :
                    # print(line)
                    # continue
                try:
                    parent_id = uid2id_map[parent[0]][0]
                    child_id = uid2id_map[child[0]][0]
                    post_time = uid2id_map[child[0]][1]
                except:
                    continue
                # print('------------')
                # print(rootindex)
                # print(child[0])
                if child[0] == rootindex :
                    # print(line)
                    continue
                # print('-+++++++:',post_time)
                # print(e)

                row.append(parent_id)
                col.append(child_id)
                # print(post_time)
                time.append(post_time)
                # print(e)
        tree = [row,col,time]
        # print(tree)

        tree, rootindex=  np.array(tree), np.array(rootindex)

        np.savez(os.path.join(rvnn_path,  dataset_name + 'matrix/' + str(id) + '.txt'), num=idx, edgeindex=tree,
                 rootindex=rootindex)
        print('ok--------------------------------')

if __name__ == '__main__':
    obj= sys.argv[1]
    get_rvnn_matrix(obj)