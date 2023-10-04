import sys, os

import torch

# sys.path.append('/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-source')
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
from PropagationNet import PropagationNet
from KnowlegeNet import KnowledgeNet
from EgoNet import EgoNet


class Net(th.nn.Module):
    def __init__(self,textCNN_param,w2vec):
        super(Net, self).__init__()
        self.KNET = KnowledgeNet(300, 64, 64, textCNN_param, w2vec).to(device)
        self.PNET = PropagationNet(5000, 64, 64).to(device)
        self.ENET = EgoNet(10, 64, 64)
        self.WK = th.nn.Linear(64, 64, bias=None)
        self.WE = th.nn.Linear(64, 64, bias=None)
        self.WKE = th.nn.Linear(64*2 , 64)

        self.WK1 = th.nn.Linear(64, 64, bias=None)
        self.WE1 = th.nn.Linear(64, 64, bias=None)
        self.WP = th.nn.Linear(64*2,64)

        self.WKP = th.nn.Linear(64*2,64)
        self.WEP = th.nn.Linear(64*2,64)
        self.WKEP = th.nn.Linear(64*2,64)

        # self.fc = th.nn.Linear(64*7, 64*4)
        self.fc1 = th.nn.Linear(64*4, 4)


    def forward(self, pdata, kdata, edata):
        kres = self.KNET(kdata)
        eres = self.ENET(edata)
        # wkres = self.WK(kres)
        # weres = self.WE(eres)

        wkeres = th.cat((kres, eres), dim=1)
        # wkeres = eres

        wkeres = self.WKE(wkeres)
        # keres = th.relu(keres)
        # wkeres = th.relu(wkeres)

        pres = self.PNET(pdata, wkeres)

        # print(pres.shape)
        # print(kres.shape)
        # print(e)c
        # print(kres)
        # print(pres)
        # x = th.cat((pres, kres, eres), 1)

        # pres = self.WP(pres)
        x = th.cat((kres,pres,eres), 1)
        # x = th.cat((kres,TD_pres,BU_pres, eres), 1)



        # wkres1 = self.WK1(wkres)
        # kpres  = th.cat((wkres1, pres), 1)
        # kpres = self.WKP(kpres)
        #
        # weres1 = self.WE1(weres)
        # epres  = th.cat((weres1, pres), 1)
        # epres = self.WEP(epres)
        #
        # kepres = th.cat((kpres,epres),1)
        # kepres = self.WKEP(kepres)

        # x = self.fc(x)
        x = self.fc1(x)

        x = F.log_softmax(x, dim=1)
        return x


def train_GCN( treeDic,x_test, x_train,PTDdroprate,PBUdroprate,kdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter):
    # raw_word_vec_file = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/w2vlstm_model/Process/word2vec_data/raw_word2vec.npy'
    raw_word_vec_file = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/hera_model_gat/Process/word2vec_data/'+dataname+'/raw_word2vec.npy'

    w2vec = np.load(raw_word_vec_file)


    # 将词向量转化为Tensor
    w2vec = torch.from_numpy(w2vec)
    # CUDA接受float32，不接受float64
    w2vec = w2vec.float()


    textCNN_param = {
        # 'vocab_size': len(word2ind),
        'embed_dim': 300,
        'class_num': 4,
        "kernel_num": 16,
        "kernel_size": [3, 4, 5],
        "dropout": 0.5,
    }
    model = Net(textCNN_param,w2vec).to(device)
    # print(w2vec.shape)
    # print(e)
    # PTD_params = list(map(id, model.PNET.TDrumorGCN.conv1.parameters()))
    # PTD_params += list(map(id, model.PNET.TDrumorGCN.conv2.parameters()))
    # KTD_params=list(map(id,model.KNET.TDrumorGCN.conv1.parameters()))
    # KTD_params += list(map(id, model.KNET.TDrumorGCN.conv2.parameters()))
    #

    # base_params=filter(lambda p:id(p) not in TD_params,model.parameters())
    base_params=model.parameters()

    optimizer = th.optim.Adam([
        {'params':base_params},
        # {'params':model.TDrumorGCN.conv1.parameters(),'lr':lr/5},
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
#def loadData(dataname,treeDic,fold_x_train,fold_x_test,knowledge_droprate,propagation_droprate):

        traindata_list, testdata_list = loadData(dataname,treeDic, x_train, x_test, propagation_droprate=PTDdroprate,\
                                                 knowledge_droprate=kdroprate)

        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_pdata,Batch_kdata,Batch_edata in tqdm_train_loader:
            Batch_pdata.to(device)
            Batch_kdata.to(device)
            Batch_edata.to(device)

            out_labels= model(Batch_pdata,Batch_kdata,Batch_edata)
            finalloss=F.nll_loss(out_labels,Batch_pdata.y)
            loss=finalloss
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_pdata.y).sum().item()
            train_acc = correct / len(Batch_pdata.y)
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
        for Batch_pdata,Batch_kdata,Batch_edata in tqdm_test_loader:
            Batch_pdata.to(device)
            Batch_kdata.to(device)
            Batch_edata.to(device)

            val_out = model(Batch_pdata,Batch_kdata,Batch_edata)
            val_loss  = F.nll_loss(val_out, Batch_pdata.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_pdata.y).sum().item()
            val_acc = correct / len(Batch_pdata.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, Batch_pdata.y)
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
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return train_losses , val_losses ,train_accs, val_accs,accs,F1,F2,F3,F4

lr=0.0005
weight_decay=1e-4
patience=10
n_epochs=200
batchsize=128
PTDdroprate=0.2
PBUdroprate=0.2
Kdroprate = 0
datasetname=sys.argv[1] #"Twitter15"、"Twitter16"
iterations=int(sys.argv[2])
model="GCN"
device = th.device('cuda:1' if th.cuda.is_available() else 'cpu')
test_accs = []
NR_F1 = []
FR_F1 = []
TR_F1 = []
UR_F1 = []

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    print("seed:", seed)

set_seed(123)

for iter in range(iterations):
    res_file = cwd+'/'+datasetname+'gatres.txt'
    f = open(res_file,'a+')
    f.write('-----------------------\n')
    fold0_x_test, fold0_x_train, \
    fold1_x_test,  fold1_x_train,  \
    fold2_x_test, fold2_x_train, \
    fold3_x_test, fold3_x_train, \
    fold4_x_test,fold4_x_train = load5foldData(datasetname)
    treeDic=loadTree(datasetname)
    train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(treeDic,
                                                                                               fold0_x_test,
                                                                                               fold0_x_train,
                                                                                               PTDdroprate,PBUdroprate,
                                                                                               Kdroprate,lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(treeDic,
                                                                                               fold1_x_test,
                                                                                               fold1_x_train,
                                                                                               PTDdroprate,PBUdroprate,
                                                                                               Kdroprate,
                                                                                               lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(treeDic,
                                                                                               fold2_x_test,
                                                                                               fold2_x_train,
                                                                                               PTDdroprate,PBUdroprate,
                                                                                               Kdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(treeDic,
                                                                                               fold3_x_test,
                                                                                               fold3_x_train,
                                                                                               PTDdroprate,PBUdroprate,
                                                                                               Kdroprate,lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(treeDic,
                                                                                               fold4_x_test,
                                                                                               fold4_x_train,
                                                                                               PTDdroprate,PBUdroprate,
                                                                                               Kdroprate,lr,
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
res = "Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
    sum(test_accs) / iterations, sum(NR_F1) / iterations, sum(FR_F1) / iterations, sum(TR_F1) / iterations,
    sum(UR_F1) / iterations)
print(str(res))
f.write(str(res))
f.close()


