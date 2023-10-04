import os
from model.hera_model.Process.dataset import GraphDataset,BiGraphDataset,UdGraphDataset
cwd=os.getcwd()



################################### load tree#####################################
def loadTree(dataname):
    if 'Twitter' in dataname:
        treePath = os.path.join(cwd,'Twitter_data/data/'+dataname+'/data.TD_RvNN.vol_5000.txt')
        print("reading twitter tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
        print('tree no:', len(treeDic))

    if dataname == "Weibo":
        treePath = os.path.join(cwd,'data/Weibo/weibotree.txt')
        print("reading Weibo tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC,Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]),line.split('\t')[3]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
        print('tree no:', len(treeDic))
    return treeDic

################################# load data ###################################
def loadData(dataname,treeDic,fold_x_train,fold_x_test,knowledge_droprate=0,propagation_droprate=0):
    pdata_path=os.path.join('/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/hera_model_gat/Twitter_data/', 'data/', dataname+'propagationgraph/')
    kdata_path=os.path.join('/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/hera_model_gat/Twitter_data/', 'data/', dataname+'knowledgegraph/')
    edata_path=os.path.join('/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/hera_model_gat/Twitter_data/', 'data/', dataname+'egograph/')

    print("loading train set", )
    traindata_list = GraphDataset(fold_x_train,treeDic, knowledge_droprate=knowledge_droprate,\
                                  propagation_droprate=propagation_droprate,knowledge_data_path= kdata_path,\
                                    propagation_data_path = pdata_path,ego_data_path = edata_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = GraphDataset(fold_x_test,treeDic, knowledge_droprate=knowledge_droprate,\
                                  propagation_droprate=propagation_droprate,knowledge_data_path= kdata_path,\
                                    propagation_data_path = pdata_path,ego_data_path = edata_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list




def loadUdData(dataname, treeDic,fold_x_train,fold_x_test,droprate):

    data_path=os.path.join('/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/Twitter_data/', 'data/',dataname+'graph')
    print("loading train set", )
    traindata_list = UdGraphDataset(fold_x_train, treeDic, droprate=droprate,data_path= data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = UdGraphDataset(fold_x_test, treeDic,data_path= data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadBiData(dataname, fold_x_train, fold_x_test, TDdroprate,BUdroprate):
    data_path = os.path.join('/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/Twitter_data/','data/', dataname + 'graph')
    print("loading train set", )
    traindata_list = BiGraphDataset(fold_x_train, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = BiGraphDataset(fold_x_test, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list



