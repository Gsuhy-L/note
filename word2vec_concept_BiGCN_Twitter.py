# import sys,os
#
# import torch
#
# sys.path.append('/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-source')
# from Process.process import *
# import torch as th
# from torch_scatter import scatter_mean
# import torch.nn.functional as F
# import numpy as np
# from tools.earlystopping import EarlyStopping
# from torch_geometric.data import DataLoader
# from tqdm import tqdm
# from Process.rand5fold import *
# from tools.evaluate import *
# from torch_geometric.nn import GCNConv,Linear
# import copy
# from torch import nn
# from cnn_model import textCNN, LSTMModel
# from lstm_config import Config
# from lcat_model.lcat.pyg import GATv1Layer, GATv2Layer
# from  build_graph import build_graph_twitter
# from build_graph import build_graph_twitter as build_3
#
#
# project_name = 'word2vec_model_concept_sim'
# datasetname = sys.argv[1]  # "Twitter15"、"Twitter16"
# iterations = int(sys.argv[2])
# model = "GCN"
# device = th.device('cuda:1' if th.cuda.is_available() else 'cpu')
# cwd = os.getcwd()
#
#
#
# class TDrumorGCN(th.nn.Module):
#     def __init__(self,in_feats,hid_feats,out_feats):
#         super(TDrumorGCN, self).__init__()
#         self.conv1 = GCNConv(in_feats, hid_feats)
#         self.conv2 = GCNConv(hid_feats, out_feats)
#         # self.conv1 = GATv1Layer(in_channels=in_feats,out_channels=hid_feats,heads=4,add_self_loops=True,bias=True,mode='lcat',share_weights_score=True,share_weights_value=True)
#         # self.conv2 = GATv1Layer(in_channels=hid_feats,out_channels=out_feats,heads=4,add_self_loops=True,bias=True,mode='lcat',share_weights_score=True,share_weights_value=True)
#
#
#
#     def forward(self, data, embedding):
#         edge_index, edge_value =  data.edge_index, data.edge_value
#         feature_ids = data.feature_ids.reshape(1,-1)
#         feature = embedding(feature_ids)
#         # x1=copy.copy(x.float())
#         # x1 = copy.copy(source_x.float())
#         # print(x)
#         # x2=copy.copy(x)
#         #rootindex = data.rootindex
#         # print(x1.shape)
#         # root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
#         # batch_size = max(data.batch) + 1
#         # for num_batch in range(batch_size):
#         #     index = (th.eq(data.batch, num_batch))
#         #     # root_extend[index] = x1[rootindex[num_batch]]
#         #     root_extend[index] = x1[num_batch]
#         # print(root_extend.shape)
#         # print(e)
#
#         # x = th.cat((feature[0],root_extend), 1)
#         # x = th.cat((feature[0],root_extend), 1)
#
#         # print(x.shape)
#
#         x = self.conv1(feature[0], edge_index = edge_index, edge_weight = edge_value)
#
#         x = F.relu(x)
#         # x = F.dropout(x, training=self.training)
#         # # print(x.shape)
#         # x = self.conv2(x, edge_index, edge_value)
#         # x = F.relu(x)
#         # root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
#         # for num_batch in range(batch_size):
#         #     index = (th.eq(data.batch, num_batch))
#         #     root_extend[index] = x2[rootindex[num_batch]]
#         # x = th.cat((x,root_extend), 1)
#         x= scatter_mean(x, data.batch, dim=0)
#
#         return x
#
# class BUrumorGCN(th.nn.Module):
#     def __init__(self,in_feats,hid_feats,out_feats):
#         super(BUrumorGCN, self).__init__()
#         self.conv1 = GCNConv(in_feats, hid_feats)
#         self.conv2 = GCNConv(hid_feats+in_feats, out_feats)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.BU_edge_index
#         x1 = copy.copy(x.float())
#         x = self.conv1(x, edge_index)
#         x2 = copy.copy(x)
#
#         rootindex = data.rootindex
#         root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
#         batch_size = max(data.batch) + 1
#         for num_batch in range(batch_size):
#             index = (th.eq(data.batch, num_batch))
#             root_extend[index] = x1[rootindex[num_batch]]
#         x = th.cat((x,root_extend), 1)
#
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
#         for num_batch in range(batch_size):
#             index = (th.eq(data.batch, num_batch))
#             root_extend[index] = x2[rootindex[num_batch]]
#         x = th.cat((x,root_extend), 1)
#
#         x= scatter_mean(x, data.batch, dim=0)
#         return x
#
# class Net(th.nn.Module):
#     def __init__(self,in_feats,hid_feats,out_feats,textCNN_param, word2vec):
#         super(Net, self).__init__()
#         self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
#         # self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
#         # self.mlp = Linear(768,64)
#         self.embedding = nn.Embedding.from_pretrained(word2vec)  # 读取预训练好的参数
#         self.embedding.weight.requires_grad = True
#         self.lstm = LSTMModel(Config.vocab_size, Config.embedding_dim, word2vec, Config.update_w2v,
#                            Config.hidden_dim, Config.num_layers, Config.drop_keep_prob, Config.n_class,
#                            Config.bidirectional)
#         self.cnn = textCNN(textCNN_param)
#         self.fc=th.nn.Linear((out_feats),4)
#
#     def forward(self, data):
#         # print(data.edge_value.dtype)
#         # print(e)
#         # print(data['doc_array'])
#         # print(data['doc_array'].shape)
#         # print(data['doc_array'])
#         # print()
#         # print()
#
#         # source_x = self.embedding(data['doc_array'])
#
#         # print(source_x.shape)
#         # print(source_x.shape)
#         # source_x = self.cnn(data['doc_array'],self.embedding)
#         # source_x = self.lstm(data['doc_array'], self.embedding)
#         # source_x = self.lstm(source_x,)\
#
#         TD_x = self.TDrumorGCN(data, self.embedding)
#         # print(data.doc_array.shape)
#         # POST_x = self.mlp(data.post_x)
#         # BU_x = self.BUrumorGCN(data)
#         # x = th.cat((TD_x,POST_x), 1)
#         # x = th.cat((TD_x,))
#         x=self.fc(TD_x)
#         x = F.log_softmax(x, dim=1)
#         return x
#
#
# def train_GCN( x_test, x_train,TDdroprate,BUdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter):
#     # raw_word_vec_file = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/w2vlstm_model/Process/word2vec_data/raw_word2vec.npy'
#     raw_word_vec_file = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/'+project_name+'/Process/word2vec_data/'+dataname+'/raw_word2vec.npy'
#
#     w2vec = np.load(raw_word_vec_file)
#
#
#     # 将词向量转化为Tensor
#     w2vec = torch.from_numpy(w2vec)
#     # CUDA接受float32，不接受float64
#     w2vec = w2vec.float()
#
#     print(w2vec.shape)
#     textCNN_param = {
#         # 'vocab_size': len(word2ind),
#         'embed_dim': 300,
#         'class_num': 4,
#         "kernel_num": 16,
#         "kernel_size": [3, 4, 5],
#         "dropout": 0.5,
#     }
#     model = Net(300,64,64,textCNN_param,w2vec).to(device)
#     TD_params=list(map(id,model.TDrumorGCN.conv1.parameters()))
#     # TD_params += list(map(id, model.TDrumorGCN.conv2.parameters()))
#     base_params=filter(lambda p:id(p) not in TD_params,model.parameters())
#     optimizer = th.optim.Adam([
#         {'params':base_params},
#         {'params':model.TDrumorGCN.conv1.parameters(),'lr':lr/5},
#         # {'params': model.TDrumorGCN.conv2.parameters(), 'lr': lr/5}
#     ], lr=lr, weight_decay=weight_decay)
#     model.train()
#     train_losses = []
#     val_losses = []
#     train_accs = []
#     val_accs = []
#     early_stopping = EarlyStopping(patience=patience, verbose=True)
#
#     for epoch in range(n_epochs):
#         # traindata_list, testdata_list = loadBiData(dataname, x_train, x_test, TDdroprate,BUdroprate)
#
#         traindata_list, testdata_list = loadData(dataname, x_train, x_test, TDdroprate)
#         train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
#         test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
#         avg_loss = []
#         avg_acc = []
#         batch_idx = 0
#         tqdm_train_loader = tqdm(train_loader)
#         for Batch_data in tqdm_train_loader:
#             Batch_data.to(device)
#             out_labels= model(Batch_data)
#             finalloss=F.nll_loss(out_labels,Batch_data.y)
#             loss=finalloss
#             optimizer.zero_grad()
#             loss.backward()
#             avg_loss.append(loss.item())
#             optimizer.step()
#             _, pred = out_labels.max(dim=-1)
#             correct = pred.eq(Batch_data.y).sum().item()
#             train_acc = correct / len(Batch_data.y)
#             avg_acc.append(train_acc)
#             print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
#                                                                                                  loss.item(),
#                                                                                                  train_acc))
#             batch_idx = batch_idx + 1
#
#         train_losses.append(np.mean(avg_loss))
#         train_accs.append(np.mean(avg_acc))
#
#         temp_val_losses = []
#         temp_val_accs = []
#         temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
#         temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
#         temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
#         temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
#         model.eval()
#         tqdm_test_loader = tqdm(test_loader)
#         for Batch_data in tqdm_test_loader:
#             Batch_data.to(device)
#             val_out = model(Batch_data)
#             val_loss  = F.nll_loss(val_out, Batch_data.y)
#             temp_val_losses.append(val_loss.item())
#             _, val_pred = val_out.max(dim=1)
#             correct = val_pred.eq(Batch_data.y).sum().item()
#             val_acc = correct / len(Batch_data.y)
#             Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
#                 val_pred, Batch_data.y)
#             temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
#                 Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
#             temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
#                 Recll2), temp_val_F2.append(F2), \
#             temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
#                 Recll3), temp_val_F3.append(F3), \
#             temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
#                 Recll4), temp_val_F4.append(F4)
#             temp_val_accs.append(val_acc)
#         val_losses.append(np.mean(temp_val_losses))
#         val_accs.append(np.mean(temp_val_accs))
#         print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
#                                                                            np.mean(temp_val_accs)))
#
#         res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
#                'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
#                                                        np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
#                'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
#                                                        np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
#                'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
#                                                        np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
#                'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
#                                                        np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
#         print('results:', res)
#         early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
#                        np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN', dataname)
#         accs =np.mean(temp_val_accs)
#         F1 = np.mean(temp_val_F1)
#         F2 = np.mean(temp_val_F2)
#         F3 = np.mean(temp_val_F3)
#         F4 = np.mean(temp_val_F4)
#         if early_stopping.early_stop:
#             print("Early stopping")
#             accs=early_stopping.accs
#             F1=early_stopping.F1
#             F2 = early_stopping.F2
#             F3 = early_stopping.F3
#             F4 = early_stopping.F4
#             break
#     return train_losses , val_losses ,train_accs, val_accs,accs,F1,F2,F3,F4
#
#
#
# def run_main(sim):
#     res_file = cwd+'/'+datasetname+'res.txt'
#     f = open(res_file,'a+')
#     f.write('-----------------------\n')
#
#
#     lr=0.0005
#     weight_decay=1e-4
#     patience=10
#     n_epochs=200
#     batchsize=128
#     TDdroprate=0
#     BUdroprate=0
#
#
#     test_accs = []
#     NR_F1 = []
#     FR_F1 = []
#     TR_F1 = []
#     UR_F1 = []
#     for iter in range(iterations):
#         fold0_x_test, fold0_x_train, \
#         fold1_x_test,  fold1_x_train,  \
#         fold2_x_test, fold2_x_train, \
#         fold3_x_test, fold3_x_train, \
#         fold4_x_test,fold4_x_train = load5foldData(datasetname)
#         # treeDic=loadTree(datasetname)
#         train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(
#                                                                                                    fold0_x_test,
#                                                                                                    fold0_x_train,
#                                                                                                    TDdroprate,BUdroprate,
#                                                                                                    lr, weight_decay,
#                                                                                                    patience,
#                                                                                                    n_epochs,
#                                                                                                    batchsize,
#                                                                                                    datasetname,
#                                                                                                    iter)
#         train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(
#                                                                                                    fold1_x_test,
#                                                                                                    fold1_x_train,
#                                                                                                    TDdroprate,BUdroprate, lr,
#                                                                                                    weight_decay,
#                                                                                                    patience,
#                                                                                                    n_epochs,
#                                                                                                    batchsize,
#                                                                                                    datasetname,
#                                                                                                    iter)
#         train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(
#                                                                                                    fold2_x_test,
#                                                                                                    fold2_x_train,
#                                                                                                    TDdroprate,BUdroprate, lr,
#                                                                                                    weight_decay,
#                                                                                                    patience,
#                                                                                                    n_epochs,
#                                                                                                    batchsize,
#                                                                                                    datasetname,
#                                                                                                    iter)
#         train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(
#                                                                                                    fold3_x_test,
#                                                                                                    fold3_x_train,
#                                                                                                    TDdroprate,BUdroprate, lr,
#                                                                                                    weight_decay,
#                                                                                                    patience,
#                                                                                                    n_epochs,
#                                                                                                    batchsize,
#                                                                                                    datasetname,
#                                                                                                    iter)
#         train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(
#                                                                                                    fold4_x_test,
#                                                                                                    fold4_x_train,
#                                                                                                    TDdroprate,BUdroprate, lr,
#                                                                                                    weight_decay,
#                                                                                                    patience,
#                                                                                                    n_epochs,
#                                                                                                    batchsize,
#                                                                                                    datasetname,
#                                                                                                    iter)
#         test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
#         NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
#         FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
#         TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
#         UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
#     res = "Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
#         sum(test_accs) / iterations, sum(NR_F1) /iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations)
#     print(str(sim)+str(res))
#     f.write(str(sim)+str(res))
#     f.close()
#
#
# def set_seed(seed=1):
#         random.seed(seed)
#         np.random.seed(seed)
#         th.manual_seed(seed)
#         th.cuda.manual_seed(seed)
#         th.backends.cudnn.deterministic = True
#         print("seed:", seed)
# # for sim in np.arange(0.1, 1.01, 0.1):
#
#
#
# set_seed(123)
#
# # build_graph_twitter(project_name,datasetname,sim=0.0)
# # build(project_name,datasetname,sim=1)
# #
# # run_main(sim=0.0)
#
#
#
# for sim_con in np.arange(0.1, 0.11, 0.1):
#     for sim_ckc in np.arange(0.1, 0.11, 0.1):
#         set_seed(123)
#
# # for i in range(10):
#         build_graph_twitter(project_name,datasetname,sim_con,sim_ckc)
#
#         run_main(str(sim_con)+str(sim_ckc))

import sys,os

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
from torch_geometric.nn import GCNConv,Linear
import copy
from torch import nn
from cnn_model import textCNN, LSTMModel
from lstm_config import Config
from lcat_model.lcat.pyg import GATv1Layer, GATv2Layer
from  build_graph import build_graph_twitter
from build_graph import build_graph_twitter as build_3


project_name = 'word2vec_model_concept_sim'
datasetname = sys.argv[1]  # "Twitter15"、"Twitter16"
iterations = int(sys.argv[2])
model = "GCN"
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
cwd = os.getcwd()



class TDrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, out_feats)
        # self.conv1 = GATv1Layer(in_channels=in_feats,out_channels=hid_feats,heads=4,add_self_loops=True,bias=True,mode='lcat',share_weights_score=True,share_weights_value=True)
        # self.conv2 = GATv1Layer(in_channels=hid_feats,out_channels=out_feats,heads=4,add_self_loops=True,bias=True,mode='lcat',share_weights_score=True,share_weights_value=True)



    def forward(self, data, embedding):
        edge_index, edge_value =  data.edge_index, data.edge_value
        feature_ids = data.feature_ids.reshape(1,-1)
        feature = embedding(feature_ids)
        # x1=copy.copy(x.float())
        # x1 = copy.copy(source_x.float())
        # print(x)
        # x2=copy.copy(x)
        #rootindex = data.rootindex
        # print(x1.shape)
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

        x = self.conv1(feature[0], edge_index = edge_index, edge_weight = edge_value)

        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # # print(x.shape)
        # x = self.conv2(x, edge_index, edge_value)
        # x = F.relu(x)
        # root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        # for num_batch in range(batch_size):
        #     index = (th.eq(data.batch, num_batch))
        #     root_extend[index] = x2[rootindex[num_batch]]
        # x = th.cat((x,root_extend), 1)
        x= scatter_mean(x, data.batch, dim=0)

        return x

class BUrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

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
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x= scatter_mean(x, data.batch, dim=0)
        return x

class Net(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,textCNN_param, word2vec):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        # self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        # self.mlp = Linear(768,64)
        self.embedding = nn.Embedding.from_pretrained(word2vec)  # 读取预训练好的参数
        self.embedding.weight.requires_grad = True
        self.lstm = LSTMModel(Config.vocab_size, Config.embedding_dim, word2vec, Config.update_w2v,
                           Config.hidden_dim, Config.num_layers, Config.drop_keep_prob, Config.n_class,
                           Config.bidirectional)
        self.cnn = textCNN(textCNN_param)
        self.fc=th.nn.Linear((out_feats),4)

    def forward(self, data):
        # print(data.edge_value.dtype)
        # print(e)
        # print(data['doc_array'])
        # print(data['doc_array'].shape)
        # print(data['doc_array'])
        # print()
        # print()

        # source_x = self.embedding(data['doc_array'])

        # print(source_x.shape)
        # print(source_x.shape)
        # source_x = self.cnn(data['doc_array'],self.embedding)
        # source_x = self.lstm(data['doc_array'], self.embedding)
        # source_x = self.lstm(source_x,)\

        TD_x = self.TDrumorGCN(data, self.embedding)
        # print(data.doc_array.shape)
        # POST_x = self.mlp(data.post_x)
        # BU_x = self.BUrumorGCN(data)
        # x = th.cat((TD_x,POST_x), 1)
        # x = th.cat((TD_x,))
        x=self.fc(TD_x)
        x = F.log_softmax(x, dim=1)
        return x


def train_GCN( x_test, x_train,TDdroprate,BUdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter):
    # raw_word_vec_file = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Knowledge_CKC_Bigcn/model/w2vlstm_model/Process/word2vec_data/raw_word2vec.npy'
    raw_word_vec_file = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Knowledge_CKC_Bigcn/model/'+project_name+'/Process/word2vec_data/'+dataname+'/raw_word2vec.npy'


    w2vec = np.load(raw_word_vec_file)


    # 将词向量转化为Tensor
    w2vec = torch.from_numpy(w2vec)
    # CUDA接受float32，不接受float64
    w2vec = w2vec.float()

    print(w2vec.shape)
    textCNN_param = {
        # 'vocab_size': len(word2ind),
        'embed_dim': 300,
        'class_num': 4,
        "kernel_num": 16,
        "kernel_size": [3, 4, 5],
        "dropout": 0.5,
    }
    model = Net(300,64,64,textCNN_param,w2vec).to(device)
    TD_params=list(map(id,model.TDrumorGCN.conv1.parameters()))
    # TD_params += list(map(id, model.TDrumorGCN.conv2.parameters()))
    base_params=filter(lambda p:id(p) not in TD_params,model.parameters())
    optimizer = th.optim.Adam([
        {'params':base_params},
        {'params':model.TDrumorGCN.conv1.parameters(),'lr':lr/5},
        # {'params': model.TDrumorGCN.conv2.parameters(), 'lr': lr/5}
    ], lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(n_epochs):
        # traindata_list, testdata_list = loadBiData(dataname, x_train, x_test, TDdroprate,BUdroprate)

        traindata_list, testdata_list = loadData(dataname, x_train, x_test, TDdroprate)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        # print(test_loader)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            out_labels= model(Batch_data)
            finalloss=F.nll_loss(out_labels,Batch_data.y)
            loss=finalloss
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        temp_full = []
        temp_doc_id = []
        temp_doc_y = []
        temp_doc_pre = []
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            val_out = model(Batch_data)
            temp_doc_id.append(Batch_data.doc_id)
            temp_doc_y.append(Batch_data.y)

            val_loss  = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            temp_doc_pre.append(val_pred)

            #因该吧iD也获取
            correct = val_pred.eq(Batch_data.y).sum().item()

            val_acc = correct / len(Batch_data.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)

        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN', dataname)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            temp_full=[temp_doc_id,temp_doc_y,temp_doc_pre]
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return train_losses , val_losses ,train_accs, val_accs,accs,F1,F2,F3,F4,temp_full



def run_main(sim):
    res_file = cwd+'/'+datasetname+'res.txt'


    f = open(res_file,'a+')
    f.write('-----------------------\n')


    lr=0.0005
    weight_decay=1e-4
    patience=10
    n_epochs=200
    batchsize=128
    TDdroprate=0
    BUdroprate=0


    test_accs = []
    NR_F1 = []
    FR_F1 = []
    TR_F1 = []
    UR_F1 = []

    for iter in range(iterations):
        fold0_x_test, fold0_x_train, \
        fold1_x_test,  fold1_x_train,  \
        fold2_x_test, fold2_x_train, \
        fold3_x_test, fold3_x_train, \
        fold4_x_test,fold4_x_train = load5foldData(datasetname)
        # treeDic=loadTree(datasetname)
        temp_full = []
        train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0,temp_full0 = train_GCN(
                                                                                                   fold0_x_test,
                                                                                                   fold0_x_train,
                                                                                                   TDdroprate,BUdroprate,
                                                                                                   lr, weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter)
        train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1,temp_full1 = train_GCN(
                                                                                                   fold1_x_test,
                                                                                                   fold1_x_train,
                                                                                                   TDdroprate,BUdroprate, lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter)
        train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2,temp_full2 = train_GCN(
                                                                                                   fold2_x_test,
                                                                                                   fold2_x_train,
                                                                                                   TDdroprate,BUdroprate, lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter)
        train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3,temp_full3 = train_GCN(
                                                                                                   fold3_x_test,
                                                                                                   fold3_x_train,
                                                                                                   TDdroprate,BUdroprate, lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter)
        train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4,temp_full4 = train_GCN(
                                                                                                   fold4_x_test,
                                                                                                   fold4_x_train,
                                                                                                   TDdroprate,BUdroprate, lr,
                                                                                                   weight_decay,
                                                                                                   patience,
                                                                                                   n_epochs,
                                                                                                   batchsize,
                                                                                                   datasetname,
                                                                                                   iter)
        test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
        NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
        FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
        TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
        UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
    temp_full = [temp_full0,temp_full1,temp_full2,temp_full3,temp_full4]
    res = "Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
        sum(test_accs) / iterations, sum(NR_F1) /iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations)
    print(str(sim)+str(res))
    f.write(str(sim)+str(res))
    f.close()

    acc_label_file = cwd+'/'+datasetname+'acc_label_without2.txt'
    f1 = open(acc_label_file,'a+')
    f1.write('-----------------------\n')
    f1.write(str(temp_full))
    f1.close()


def set_seed(seed=1):
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.backends.cudnn.deterministic = True
        print("seed:", seed)
# for sim in np.arange(0.1, 1.01, 0.1):



set_seed(123)

# build_graph_twitter(project_name,datasetname,sim=0.0)
# build(project_name,datasetname,sim=1)
#
# run_main(sim=0.0)



for sim_con in np.arange(1.1, 1.11, 0.1):
    for sim_ckc in np.arange(1.1, 1.11, 0.1):

        set_seed(123)

# for i in range(10):
        build_graph_twitter(project_name,datasetname,sim_con,sim_ckc)

        run_main(str(sim_con)+str(sim_ckc))