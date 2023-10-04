import sys,os
sys.path.append('/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-source')
from Process.process import *
import torch as th
from torch_scatter import scatter_mean

import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv,GATConv
import copy
from lcat_model.lcat.pyg import GATv1Layer, GATv2Layer

device = th.device('cuda:1' if th.cuda.is_available() else 'cpu')



class TDrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(TDrumorGCN, self).__init__()
        # self.conv1 = GCNConv(in_feats, hid_feats)
        # self.conv2 = GCNConv(hid_feats+in_feats, out_feats)
        # self.conv2 = GCNConv(hid_feats, out_feats)

        # self.conv1 = GATv1Layer(in_feats+64,hid_feats,heads=4,concat=False)
        # self.conv2 = GATv1Layer(hid_feats,out_feats,heads=4,concat=False)
        self.conv1 = GATv1Layer(in_channels=in_feats+64,out_channels=hid_feats,heads=4,add_self_loops=True,bias=True,mode='lcat',share_weights_score=True,share_weights_value=True)
        self.conv2 = GATv1Layer(in_channels=hid_feats+in_feats+64,out_channels=out_feats,heads=4,add_self_loops=True,bias=True,mode='lcat',share_weights_score=True,share_weights_value=True)

    def forward(self, data, keres):
        # print(data.shape)
        # print(keres.shape)
        # print
        x, edge_index = data.x, data.edge_index

        # print(x.shape)
        ke_x=copy.copy(keres.float())
        # print(x.shape)
        # x2=copy.copy(x)
        rootindex = data.rootindex
        # print(rootindex)
        root_extend = th.zeros(len(data.batch), ke_x.size(1)).to(device)
        batch_size = max(data.batch) + 1
        # print(batch_size)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            # print(rootindex[num_batch])
            # root_extend[index] = ke_x[rootindex[num_batch]]
            root_extend[index] = ke_x[num_batch]

        x = th.cat((x,root_extend), 1)
        x1 = copy.copy(x)

        x = self.conv1(x, edge_index)

        # print(x.shape)
        x2=copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # x = x + x1

        x = self.conv2(x, edge_index)
        # x = th.add(x,x2)

        # x2 = copy.copy(x)
        # x = x + x2
        # root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        # for num_batch in range(batch_size):
        #     index = (th.eq(data.batch, num_batch))
        #     root_extend[index] = x2[rootindex[num_batch]]
        # x = th.cat((x,root_extend), 1)
        x = th.add(x, x2)

        x = F.relu(x)
        x= scatter_mean(x, data.batch, dim=0)

        return x

class BUrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BUrumorGCN, self).__init__()
        # self.conv1 = GATConv(in_feats+64, hid_feats ,heads=4,concat=False)
        # self.conv2 = GATConv(hid_feats, out_feats, heads=4,concat=False)
        # self.conv2 = GCNConv(hid_feats, out_feats)
        """#         in_channels=in_channels,
#         out_channels=out_channels,
#         negative_slope=0.2,
#         add_self_loops=add_self_loops,
#         heads=heads,
#         bias=True,
#         mode=mode,
#         share_weights_score=False,
#         share_weights_value=False,
#         aggr='mean',"""
        self.conv1 = GATv1Layer(in_channels=in_feats+64,out_channels=hid_feats,heads=4,add_self_loops=True,bias=True,mode='lcat',share_weights_score=True,share_weights_value=True)
        self.conv2 = GATv1Layer(in_channels=hid_feats+in_feats+64,out_channels=out_feats,heads=4,add_self_loops=True,bias=True,mode='lcat',share_weights_score=True,share_weights_value=True)


    def forward(self, data, keres):
        # x, edge_index = data.x, data.BU_edge_index
        x, edge_index = data.x, data.BU_edge_index

        # print(x.shape)
        # x2=copy.copy(x)
        ke_x = copy.copy(keres.float())

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), ke_x.size(1)).to(device)
        batch_size = max(data.batch) + 1

        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = ke_x[num_batch]

        x = th.cat((x, root_extend), 1)
        x1 = copy.copy(x.float())


        x = self.conv1(x, edge_index)

        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # x = th.add(x,x2)

        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        # x = th.cat((x,root_extend), 1)
        x = th.add(x,x2)
        x = F.relu(x)
        x= scatter_mean(x, data.batch, dim=0)
        return x

class PropagationNet(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(PropagationNet, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        # self.fc=th.nn.Linear((out_feats+hid_feats)*2,4)
        self.fc=th.nn.Linear((out_feats)*2,4)

    def forward(self, data, keres):
        TD_x = self.TDrumorGCN(data, keres)
        BU_x = self.BUrumorGCN(data, keres)
        x = th.cat((BU_x,TD_x), 1)
        # x=self.fc(x)
        # x = F.log_softmax(x, dim=1)
        return x