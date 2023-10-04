import sys, os

import torch

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
from torch_geometric.nn import GCNConv, Linear,GATConv
import copy
from torch import nn
from cnn_model import textCNN, LSTMModel
from lstm_config import Config
from lcat_model.lcat.pyg import GATv1Layer, GATv2Layer

# device =
device = th.device('cuda:1' if th.cuda.is_available() else 'cpu')

class TDrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GATv1Layer(in_channels=in_feats,out_channels=hid_feats,heads=4,add_self_loops=True,bias=True,mode='lcat',share_weights_score=True,share_weights_value=True)
        self.conv2 = GATv1Layer(in_channels=hid_feats+in_feats,out_channels=hid_feats,heads=4,add_self_loops=True,bias=True,mode='lcat',share_weights_score=True,share_weights_value=True)
        # self.conv1 = GATConv(in_feats, hid_feats, heads=4,concat=False)
        # self.conv2 = GATConv(hid_feats + in_feats, out_feats,heads=4,concat=False)
        # self.conv1 = GATConv(in_feats+64, hid_feats ,heads=4,concat=False)
        # self.conv2 = GATConv(hid_feats, out_feats, heads=4,concat=False)
        # self.conv1 = GCNConv(in_feats, out_feats)
        # self.conv2 = GCNConv(hid_feats+in_feats, out_feats)
        self.droupout_rate = 0.2
        # self.w1 = th.nn.Linear(hid_feats , hid_feats )
        self.dropout = th.nn.Dropout(self.droupout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x)
        # x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, p=self.droupout_rate, training=self.training)

        x = self.conv2(x, edge_index)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]

        # x = th.cat((x,root_extend), 1)
        x = th.add(x,x2)

        x = F.relu(x)
        # x2 = copy.copy(x)
        # x = x+x2

        # x = scatter_mean(F.relu(self.w1(x)), data.batch, dim=0)
        x = scatter_mean(x, data.batch, dim=0)

        return x

class EgoNet(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(EgoNet, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        # self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        # self.fc=th.nn.Linear((out_feats+hid_feats)*2,4)
        self.fc=th.nn.Linear((out_feats),4)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        # BU_x = self.BUrumorGCN(data)
        # x = th.cat((BU_x,TD_x), 1)
        # x=self.fc(x)
        # x = F.log_softmax(x, dim=1)
        return TD_x