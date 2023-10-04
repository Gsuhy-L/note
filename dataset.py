import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data

class GraphDataset(Dataset):
    def __init__(self, fold_x,treeDic, lower=2, upper=100000, knowledge_droprate=0,propagation_droprate = 0,
                 knowledge_data_path=os.path.join('..','..', 'data', 'Weibograph'),
                 propagation_data_path=os.path.join('..','..', 'data', 'Weibograph'),
                 ego_data_path=os.path.join('..', '..', 'data', 'Weibograph'),

                 ):
        print(len(fold_x))
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        print(len(self.fold_x))
        # self.treeDic = treeDic
        self.knowledge_data_path = knowledge_data_path
        self.knowledge_droprate = knowledge_droprate

        self.treeDic = treeDic
        self.propagation_data_path = propagation_data_path
        self.propagation_droprate = propagation_droprate

        self.ego_data_path = ego_data_path


    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        # print('----------')
        # print(len(self.fold_x))

        knowledge_data=np.load(os.path.join(self.knowledge_data_path, id + ".npz"), allow_pickle=True)

        knowledge_edgeindex = knowledge_data['edgeindex'][0]
        # print(edgeindex.toarray())
        # print(knowledge_edgeindex)

        knowledge_edgeindex , knowledge_edgevalue, knowledge_edgeshape = sparse_mx_to_torch(knowledge_edgeindex)
        if self.knowledge_droprate > 0:
            knowledge_row = list(knowledge_edgeindex[0])
            knowledge_col = list(knowledge_edgeindex[1])
            knowledge_length = len(knowledge_row)
            knowledge_poslist = random.sample(range(knowledge_length), int(knowledge_length * (1 - self.knowledge_droprate)))
            knowledge_poslist = sorted(knowledge_poslist)
            knowledge_row = list(np.array(knowledge_row)[knowledge_poslist])
            knowledge_col = list(np.array(knowledge_col)[knowledge_poslist])
            knowledge_new_edgeindex = [knowledge_row, knowledge_col]
        else:
            knowledge_new_edgeindex = knowledge_edgeindex
        # print(id)
        knowledge_new_feature_ids = knowledge_data['feature_ids'].reshape(-1,1)
        try:
            propagation_data=np.load(os.path.join(self.propagation_data_path, id + ".npz"), allow_pickle=True)
            # print('----------------------')

            propagation_edgeindex = propagation_data['edgeindex']
            if self.propagation_droprate > 0:
                propagation_row = list(propagation_edgeindex[0])
                propagation_col = list(propagation_edgeindex[1])
                propagation_length = len(propagation_row)
                propagation_poslist = random.sample(range(propagation_length), int(propagation_length * (1 - self.propagation_droprate)))
                propagation_poslist = sorted(propagation_poslist)
                propagation_row = list(np.array(propagation_row)[propagation_poslist])
                propagation_col = list(np.array(propagation_col)[propagation_poslist])
                propagation_new_edgeindex = [propagation_row, propagation_col]
            else:
                propagation_new_edgeindex = propagation_edgeindex


            propagation_burow = list(propagation_edgeindex[1])
            propagation_bucol = list(propagation_edgeindex[0])

            if self.propagation_droprate > 0:
                propagation_length = len(propagation_burow)
                propagation_poslist = random.sample(range(propagation_length), int(propagation_length * (1 - self.propagation_droprate)))
                propagation_poslist = sorted(propagation_poslist)
                propagation_row = list(np.array(propagation_burow)[propagation_poslist])
                propagation_col = list(np.array(propagation_bucol)[propagation_poslist])
                propagation_bunew_edgeindex = [propagation_row, propagation_col]
            else:
                propagation_bunew_edgeindex = [propagation_burow,propagation_bucol]
        except:
            print(id)

        user_ego = np.load(self.ego_data_path + str(id) + '.npz', allow_pickle=True)
        ego_twitter_id = id
        ego_root_feature = np.array(eval(str(user_ego['root_feature'])))
        ego_tree_feature = np.array(eval(str(user_ego['tree_feature'])))
        ego_edge_index = np.array(eval(str(user_ego['edge_index'])))
        ego_root_index = user_ego['root_index']
        # ego_user_id = user_ego[5]

        return Data(x=torch.tensor(propagation_data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(propagation_new_edgeindex),BU_edge_index=torch.LongTensor(propagation_bunew_edgeindex),
             y=torch.LongTensor([int(propagation_data['y'])]), root=torch.LongTensor(propagation_data['root']),
             rootindex=torch.LongTensor([int(propagation_data['rootindex'])])),\
               Data(x=torch.tensor(knowledge_data['x'], dtype=torch.float32),
                    feature_ids = torch.tensor(knowledge_new_feature_ids),
                    doc_array = torch.tensor(knowledge_data['doc_array']),
                    # new_x = torch.tensor(data['x'], dtype=torch.float32),
                        # post_x = torch.tensor(data['post_x'],dtype=torch.float32),
                        edge_index=torch.LongTensor(knowledge_new_edgeindex),
                        edge_value = torch.FloatTensor(knowledge_edgevalue),
                        y=torch.LongTensor([int(knowledge_data['y'])])), \
               Data(x=torch.tensor(ego_tree_feature, dtype=torch.float32),
                    edge_index=torch.LongTensor(ego_edge_index), y=torch.LongTensor([int(propagation_data['y'])]),
                    root=torch.LongTensor([ego_root_feature]),
                    rootindex=torch.LongTensor([int(ego_root_index)]),
                    tree_text_id=torch.LongTensor([int(ego_twitter_id)]))



        '''return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))'''



def sparse_mx_to_torch(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # print(type(sparse_mx))
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # print('----------------------------------------------------')
    if len(sparse_mx.row) == 0 or len(sparse_mx.col) == 0:
        print(sparse_mx.row, sparse_mx.col)
        print('data bug')
        print('sparse_mx.data',sparse_mx.data)
        print('sparse_mx.shape',sparse_mx.shape)
    # print('--------row col-------:',type(sparse_mx.row),type(sparse_mx.col)) dp.ndarray
    if np.NAN in sparse_mx.data:
        print('有NaN数据')
    # with open('test_matraix_data.txt','a',encoding='utf-8')as f:
    #     v_list = []
    #     for v in sparse_mx.data:
    #         v_list.append(str(v))
    #     f.writelines(v_list)
    # assert sparse_mx.data.sum() == np.float32(len(sparse_mx.row))
    # print('data sparse_mx.data.sum',sparse_mx.data.sum(),type(sparse_mx.data.sum()))
    # print('data len(sparse_mx.row)',len(sparse_mx.row),type(len(sparse_mx.row)))
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data.astype(np.float32))
    shape = torch.Size(sparse_mx.shape)
    return indices,values,shape

def collate_fn(data):
    return data

class BiGraphDataset(Dataset):
    def __init__(self, fold_x,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list( fold_x)
        # self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))


class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..','data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]

        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))
