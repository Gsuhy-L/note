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
from torch_geometric.nn import GCNConv, Linear
import copy
from torch import nn
from cnn_model import textCNN, LSTMModel
from lstm_config import Config


class TDrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, out_feats)

    def forward(self, data, source_x, embedding):
        edge_index, edge_value = data.edge_index, data.edge_value
        feature_ids = data.feature_ids.reshape(1, -1)
        feature = embedding(feature_ids)
        # x1=copy.copy(x.float())
        x1 = copy.copy(source_x.float())
        # print(x)
        # x2=copy.copy(x)
        # rootindex = data.rootindex
        # root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        # batch_size = max(data.batch) + 1
        # for num_batch in range(batch_size):
        #     index = (th.eq(data.batch, num_batch))
        #     # root_extend[index] = x1[rootindex[num_batch]]
        #     root_extend[index] = x1[num_batch]
        # print(root_extend.shape)
        # print(e)

        # x = th.cat((feature[0],root_extend), 1)
        # x = th.cat((feature[0],root_extend), 1)

        # print(x.shape)
        x = self.conv1(feature[0], edge_index=edge_index, edge_weight=edge_value)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_value)
        x = F.relu(x)
        # root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        # for num_batch in range(batch_size):
        #     index = (th.eq(data.batch, num_batch))
        #     root_extend[index] = x2[rootindex[num_batch]]
        # x = th.cat((x,root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)

        return x


class BUrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
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
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        return x


class KnowledgeNet(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, textCNN_param, word2vec):
        super(KnowledgeNet, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        # self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        # self.mlp = Linear(768,64)
        self.embedding = nn.Embedding.from_pretrained(word2vec)  # 读取预训练好的参数
        self.embedding.weight.requires_grad = True
        self.lstm = LSTMModel(Config.vocab_size, Config.embedding_dim, word2vec, Config.update_w2v,
                              Config.hidden_dim, Config.num_layers, Config.drop_keep_prob, Config.n_class,
                              Config.bidirectional)
        self.cnn = textCNN(textCNN_param)
        self.fc = th.nn.Linear((out_feats), 4)

    def forward(self, data):
        # print(data['doc_array'])

        source_x = self.embedding(data['doc_array'])
        TD_x = self.TDrumorGCN(data, source_x, self.embedding)
        # print(data.doc_array.shape)
        # POST_x = self.mlp(data.post_x)
        # BU_x = self.BUrumorGCN(data)
        # x = th.cat((TD_x,POST_x), 1)
        # x = th.cat((TD_x,))
        # x = self.fc(TD_x)
        # x = F.log_softmax(x, dim=1)
        return TD_x

