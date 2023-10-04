import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm
from bert_model import BertLayer
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim

# cwd=os.getcwd()
if len(sys.argv) < 2:
    sys.exit("Use: python knowledge_build_graph.py <dataset>")

# settings
# datasets = ['mr', 'ohsumed', 'R8', 'R52', 'TREC', 'ag_news', 'WebKB', 'SST1', 'SST2']

dataset = sys.argv[1]
seq_lenth = sys.argv[2]
# if dataset not in datasets:
#     sys.exit("wrong dataset name")

try:
    window_size = int(sys.argv[3])
except:
    window_size = 3
    print('using default window size = 3')

try:
    weighted_graph = bool(sys.argv[4])
except:
    weighted_graph = False
    print('using default unweighted graph')

truncate = False  # whether to truncate long document
MAX_TRUNC_LEN = 35

print('loading raw data')


doc_content_list = []
doc_name_list = []
content_path = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/hera_model/Twitter_data/data/' + dataset

# for file in os.listdir(content_path):
with open(content_path + '/clean_content.txt', 'r') as f:
    for line in f.readlines():
        # print(line)
        doc_name_list.append(line.split('\t')[0])
        doc_content_list.append(line.split('\t')[1].strip())
no_url_doc_name_list = []
# no_url_doc_content_lst = []
# with open(content_path + '/source_tweets_del_url.txt', 'r') as f:
#     for line in f.readlines():
#         # print(line)
#         no_url_doc_name_list.append(line.split('\t')[0])
#         no_url_doc_content_lst.append(line.split('\t')[1].strip())

# print(doc_name_list)
# build corpus vocabulary



# word_set = set()
# flag = 0
# for doc_words in doc_content_list:
#     flag += 1
#     words = doc_words.split()
#     word_set.update(words)
# #:判断一共有多少个单词
# vocab = list(word_set)
# print(vocab)

# print(e)
# vocab_size = len(vocab)
# #每个单词给他一个编号
# word_id_map = {}
# for i in range(vocab_size):
#     word_id_map[vocab[i]] = i

# def stopwordslist():
#     """
#     创建停用词表
#     :return:
#     """
#     stopwords = [line.strip() for line in open('word2vec_data/stopword.txt', encoding='UTF-8').readlines()]
#     return stopwords


def build_word2id(dataset):
    """
    将word2id词典写入文件中，key为word，value为索引
    :param file: word2id保存地址
    :return: None
    """
    # 加载停用词表
    # stopwords = stopwordslist()
    word2id = {'_PAD_': 0}
    # 文件路径
    #path = [Config.train_path, Config.val_path]
    # print(path)
    # 遍历训练集与验证集

    # 打开文件
    path = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/hera_model/Twitter_data/data/'+dataset+'/clean_content.txt'
    file_path = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/hera_model/Process/word2vec_data/word2id.txt'

    with open(path, encoding='utf-8') as f:
        # 遍历文件每一行
        for line in f.readlines():
            # print(line)
            out_list = []
            # 去掉首尾空格并按照空格分割
            sp = line.split('\t')[1].split()
            # 遍历文本部分每一个词
            # print(sp)
            for word in sp[:]:
                # 如果词不是停用词
                # if word not in stopwords:
                    # print(word)
                    # 在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果没有找到匹配的，则返回空列表。
                    # rt = re.findall('[a-zA-Z]+', word)
                    # # 如果word不等于制表符
                    # # print(rt)
                    # if word != '\t':
                    #     # 如果词匹配的字符串为1，则继续遍历下一个词
                    #     if len(rt) == 1:
                    #         continue
                    #     # 如果词匹配的字符串为0，则将这个词添加到out_list中
                    #     else:
                    out_list.append(word)

            # 遍历out_list中的词
            # print(out_list)
            for word in out_list:
                # 如果这些词不在word2id字典的key中,则添加到word2id字典中
                if word not in word2id.keys():
                    word2id[word] = len(word2id)
    for key in word2id:
        word2id[key] = int(word2id[key])

    # 构建id2word
    id2word = {}
    for key, val in word2id.items():
        id2word[val] = key
    # 打开输出文件并进行文件写入
    with open(file_path, 'w', encoding='utf-8') as f:
        # 遍历词典中的每一个词
        for w in word2id:
            f.write(w + '\t')
            f.write(str(word2id[w]))
            f.write('\n')
    return word2id, id2word


def build_word2vec(word2id, save_to_path=None):
    """
    使用word2vec对单词进行编码
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    # 词的总数量
    n_words = max(word2id.values()) + 1
    # 加载预训练的词向量
    glove_file = datapath('/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/hera_model/Process/word2vec_data/glove.6B.300d.txt')
    tmp_file = get_tmpfile('/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/hera_model/Process/word2vec_data/word2vec.6B.300d.txt')
    glove2word2vec(glove_file,tmp_file)
    raw_word_vec_file = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/hera_model/Process/word2vec_data/raw_word2vec'
    model = gensim.models.KeyedVectors.load_word2vec_format(tmp_file,binary=False)
    # 初始化词向量
    #此项了的唯独等于个数诚意唯独
    word_vecs = np.array(np.random.uniform(-0.01, -0.01, [n_words, model.vector_size]))
    # 遍历每个单词
    for word in word2id.keys():
        try:
            # 构建词向量
            word_vecs[word2id[word]] = model[word]
            # print()
        except KeyError:
            pass
    # 将word_vecs保存到文件中
    if save_to_path:
        # with open(raw_word_vec_file, 'w', encoding='utf-8') as f:
        #     for vec in word_vecs:
        #         vec = [str(w) for w in vec]
        #         f.write(' '.join(vec))
        #         f.write('\n')
        # with open(raw_word_vec_file, 'w') as f1:
        #     f1.write(word_vecs)
        np.save(raw_word_vec_file,word_vecs)
    # 返回word_vecs数组
    # print('word_vecs',word_vecs.shape)
    return word_vecs



def get_label(dataset):
    labelPath = os.path.join('/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/hera_model/Twitter_data/',\
                             "data/" + dataset + "/label.txt")
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']

    print("loading tree label")
    event, y = [], []
    l1 = l2 = l3 = l4 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split(':')[0], line.split(':')[1]
        label=label.lower()
        event.append(eid)
        if label in labelset_nonR:
            labelDic[eid]=0
            l1 += 1
        if label  in labelset_f:
            labelDic[eid]=1
            l2 += 1
        if label  in labelset_t:
            labelDic[eid]=2
            l3 += 1
        if label  in labelset_u:
            labelDic[eid]=3
            l4 += 1
    return labelDic
    # print(len(labelDic))


def build_graph(word_id_map, id_word_map, dataset, seq_lenth):
    # load pre-trained word embeddings
    word_embeddings_dim = 300
    word_embeddings = {}


    with open("/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/glove/" + 'glove.6B.' + str(
            word_embeddings_dim) + 'd.txt', 'r') as f:
        for line in f.readlines():
            data = line.split()
            word_embeddings[str(data[0])] = list(map(float, data[1:]))

    oov = {}
    for v in range(len(word_id_map)):
        oov[v] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)
    i=0
    for word in word_id_map.keys():

        if word in word_embeddings:
            # print()
            oov[word_id_map[word]] = word_embeddings[word]

            # print()
        else:
            # print(word)
            # print(i)
            i+=1
    labels_lst = get_label(dataset)
    x_adj = []
    x_feature = []
    # y = []
    doc_len_list = []
    vocab_set = set()
    # print(doc_name_list)
    for i in tqdm(range(len(doc_content_list))):
        doc_name = doc_name_list[i]
        # post_text = no_url_doc_content_lst[i]
        # print(doc_name)
        doc_words = doc_content_list[i].split()
        if truncate:
            doc_words = doc_words[:MAX_TRUNC_LEN]
        doc_len = len(doc_words)

        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)

        doc_len_list.append(doc_nodes)
        vocab_set.update(doc_vocab)

        doc_word_id_map = {}
        for j in range(doc_nodes):
            #在单个文件中，给每个词一个id
            doc_word_id_map[doc_vocab[j]] = j
        # sliding windows
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]
                    word_p_id = word_id_map[word_p]  # doc_word_id_map[word_p]
                    word_q = window[q]
                    word_q_id = word_id_map[word_q]  # doc_word_id_map[word_q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.

        row = []
        col = []
        weight = []
        features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            #里边存放的ID是总的次表的ID，需要把它专程单个次表的ID
            row.append(doc_word_id_map[id_word_map[p]])  # p
            col.append(doc_word_id_map[id_word_map[q]])  # q
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))



        temp_features_id = []
        feature_ids =[]
        tmp_features = []
        post_features = []
        #这里生成节点的特征。
        # print(doc_word_id_map)
        # print(e)
        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            # print(k)
            # print(word_id_map[k])
            # tmp_features.append(word_embeddings[k] if k in word_embeddings else oov[k])
            temp_features_id.append(word_id_map[k])
            tmp_features.append(oov[word_id_map[k]])
            # tmp_features.append(k)
        # print(feature_ids.dtype)
        # print(e)
        feature_ids.append(temp_features_id)
        # print('=====================')
        features = tmp_features
        # post_features.append(post_text)
        # bert_model = BertLayer(dataset)
        # features = bert_model.forward(tmp_features)
        # feature_post = bert_model.forward(post_features)


        y = labels_lst[doc_name]
        # x_adj.append(adj)
        # x_feature.append(features)


        # return None
        # features = np.array(features.cpu())
        feature_ids = np.array(feature_ids)
        features = np.array(features)
        # post_features = np.array(feature_post.cpu())
        adj_lst = []
        adj_lst.append(adj)
        adj_lst = np.array(adj_lst)
        # print(adj.toarray())
        # print(e)
        y=np.array(y)
        doc_array = prepare_data(dataset, word_id_map, doc_name, seq_lenth)
        # print(doc_array)
        # print(feature_ids)


        np.savez(os.path.join('/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/hera_model/Twitter_data/',\
                              'data/' + dataset + 'graph/' + doc_name + '.npz'), x=features, doc_array = doc_array,  edgeindex=adj_lst, \
                                  feature_ids = feature_ids, y = y)

        # data = np.load(os.path.join('/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/Twitter_data/',\
        #                       'data/' + dataset + 'graph/' + doc_name + '.npz'), allow_pickle=True)

def text_to_array(dataset, word2id, seq_lenth, doc_id):
    """
    有标签文本转为索引数字模式
    :param word2id: word2id
    :param seq_lenth: 句子最大长度
    :param path: 文件路径
    :return:
    """
    # 存储标签
    lable_array = []
    # 句子索引初始化
    i = 0
    sa = []
    path = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Twitter_knowledge_data/model/hera_model/Twitter_data/data/'+dataset+'/clean_content.txt'

    # 获取句子个数
    # print(word2id)
    # print(e)
    with open(path, encoding='utf-8') as f1:
        # 打开文件并遍历文件每一行
        for l1 in f1.readlines():
            # 返回分割后的字符串列表
            # print(l1.split('\t')[0])
            # print(e)
            s = l1.split('\t')[1].strip().split()
            # 去掉标签
            s1 = s[:]
            # 单词转索引数字
            new_s = [word2id.get(word, 0) for word in s1]
            # 存储由索引数字表示的文本列表
            # print(new_s)

            sa.append(new_s)
        # print(len(sa))

    # print(label_dic)
    content_idx_dict = {}
    with open(path, encoding='utf-8') as f:
        # 初始化句子array；行：句子个数 列：句子长度
        sentences_array = np.zeros(shape=(len(sa), seq_lenth),dtype=np.int)
        # print(sentences_array)
        # print(e)
        # 遍历每一句话
        for line in f.readlines():
            # print(line)
            sl1 = line.split('\t')[1].strip().split()
            # 去掉标签
            sen = sl1[:]
            # 单词转索引数字,不存在则为0
            new_sen = [word2id.get(word, 0) for word in sen]
            # print(new_sen)
            # 转换为(1,sen_len)
            new_sen_np = np.array(new_sen).reshape(1, -1)
            # print(new_sen_np)

            # 补齐每个句子长度，少了就直接赋值,0填在前面。
            # np.size，返回沿给定轴的元素数
            # print(np.size(new_sen_np, 1))
            if np.size(new_sen_np, 1) < seq_lenth:
                sentences_array[i, seq_lenth - np.size(new_sen_np, 1):] = new_sen_np[0, :]
            # 长了进行截断
            else:
                sentences_array[i, 0:seq_lenth] = new_sen_np[0, 0:seq_lenth]
            # print(sentences_array)
            content_idx = line.split('\t')[0]
            content_idx_dict[content_idx] = sentences_array[i]
            i = i + 1


                # 标签
                # lable = int(sl1[0])
                # lable = label_dic[line.split('\t')[0]]
                # print(line.split('\t')[0])
                # print(lable)
                # print(e)
                # lable_array.append(lable)
            # else:
                # print(e)
        doc_sentences_array=[]
        # print(doc_id)


        doc_sentences_array.append(list(content_idx_dict[doc_id]))
        # 返回索引模式的文本以及标签
        # print('===============')
        # print(doc_lst)
        # print(lable_array)
        # print(e)
        # print(doc_lst)
        # print(lable_array)
        # print(e)
        # print(doc_sentences_array)
        # print(e)
    return np.array(doc_sentences_array)

def prepare_data(dataset, w2id, doc_id, seq_lenth):
    """
    得到数字索引表示的句子和标签
    :param w2id: word2id
    :param train_path: 训练文件路径
    :param val_path: 验证文件路径
    :param test_path: 测试文件路径
    :param seq_lenth: 句子最大长度
    :return:
    """
    # 对训练集、验证集、测试集处理，将文本转化为由单词索引构成的array
    train_array = text_to_array(dataset=dataset, word2id=w2id, seq_lenth=seq_lenth, doc_id = doc_id)
    # print(train_lst)
    # print(train_lable)
    # print(e)
    # val_array = text_to_array(dataset=dataset, word2id=w2id, seq_lenth=seq_lenth, doc_id=val_lst)
    return train_array

print('building graphs for training')
word_id_map, id_word_map = build_word2id(dataset)

word2vec = build_word2vec(word_id_map,True)

build_graph(word_id_map, id_word_map, dataset, seq_lenth=25)

