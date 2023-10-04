import os
from collections import defaultdict
cwd = os.getcwd()
datasetname = 'twitter15'
without_file = cwd + '/' + datasetname + 'acc_label_without2.txt'
con_file = cwd + '/' + datasetname + 'acc_label_con.txt'
two_file = cwd + '/' + datasetname + 'acc_label_two.txt'

open_without_file = open(without_file,'r').read()
file_arr = eval(open_without_file)
# print(file)

without_set = defaultdict(list)


without_set = defaultdict(list)
for cross_num in range(len(file_arr)):
    #0-2
    cross_batch_data = file_arr[cross_num]
    #3
    # for type_idx in range(len(file_arr[cross_num])):
    # print(file_arr[cross_num][0])

    # print()

    for batch_data_num in range(len(file_arr[cross_num][0])):
        # print(file_arr[cross_num][0][batch_data_num])
        for batch_idx in range(len(file_arr[cross_num][0][batch_data_num])):
            # print(batch_idx)
            # print(file_arr[cross_num][0][batch_data_num][batch_idx])
            # for idx in range(len(file_arr[cross_num][0][batch_data_num][batch_idx])):
                doc_id = file_arr[cross_num][0][batch_data_num][batch_idx]
                # doc_y =
                doc_y = file_arr[cross_num][1][batch_data_num][batch_idx]
                doc_pre = file_arr[cross_num][2][batch_data_num][batch_idx]

                # if doc_y != doc_pre:
                    # print(doc_id)
                without_set[doc_id].append([doc_y,doc_pre])

# print(without_set)
con_set = defaultdict(list)
for cross_num in range(len(file_arr)):
    # 0-2
    cross_batch_data = file_arr[cross_num]
    # 3
    # for type_idx in range(len(file_arr[cross_num])):
    # print(file_arr[cross_num][0])

    # print()

    for batch_data_num in range(len(file_arr[cross_num][0])):
        # print(file_arr[cross_num][0][batch_data_num])
        for batch_idx in range(len(file_arr[cross_num][0][batch_data_num])):
            # print(batch_idx)
            # print(file_arr[cross_num][0][batch_data_num][batch_idx])
            # for idx in range(len(file_arr[cross_num][0][batch_data_num][batch_idx])):
            doc_id = file_arr[cross_num][0][batch_data_num][batch_idx]
            # doc_y =
            doc_y = file_arr[cross_num][1][batch_data_num][batch_idx]
            doc_pre = file_arr[cross_num][2][batch_data_num][batch_idx]

            if (doc_y != doc_pre) and (doc_id in without_set):
                # print(doc_id)
                con_set[doc_id].append([doc_y, doc_pre])
print(len(con_set))
#defaultdict(<class 'list'>, {650952376954650629: [[3, 3]], 562313802369073153: [[2, 2]], 553960736964476928: [[1, 1]], 692142338890661888: [[0, 0]], 517712193841037312: [[3, 3]], 767710245779103744: [[0, 0]], 693485676881403905: [[0, 0]], 505657661120348163: [[1, 1]], 692566765822435328: [[0, 0]], 531525016794697729: [[3, 3]], 687766167558164481: [[0, 0]], 554655549896159233: [[1, 1]], 765141361033109504: [[0, 0]], 693921710383337472: [[0, 0]], 501934077612941312: [[3, 3]]})
#现在我需要列出所有

# content_file = c
content_path = cwd+ '/Twitter_data/data/' + 'twitter15' + '/clean_content.txt'

content_set = defaultdict(list)
with open(content_path, encoding='utf-8') as f:
    # 遍历文件每一行
    for line in f.readlines():
        # print(line)
        out_list = []
        # 去掉首尾空格并按照空格分割
        sp_id = line.strip('\n').split('\t')[0]
        sp = line.split('\t')[1].split()
        # print()
        # print(sp_id)
        # print(sp)
        content_set[sp_id]=sp
# print(content_set)
doc_concept_entity = defaultdict(list)
doc_entity = defaultdict(list)

with open(cwd +'/Twitter_data/data/' + 'twitter15' + '/sig_concept.txt', 'r') as f_c:
    for concept_line in f_c.readlines():
        # TEXT
        c_id = concept_line.strip('\n').split('\t')[0]
        # print(line)
        # SET
        e_set = eval(concept_line.strip('\n').split('\t')[1])
        if len(e_set) != 0:
            for k, c_set in e_set.items():
                doc_entity[c_id].append(k)
                # if len(c_set) != 0:
                #     for concept in c_set:
                if e_set not in doc_concept_entity[c_id]:
                    doc_concept_entity[c_id].append(e_set)

                        # con_sim = cos_sim(w2vec[word_id_map[k]],w2vec[word_id_map[concept]])

ckc_path = cwd+'/Twitter_data'+"/data/"+'twitter15'+'/bilearavg_condidate_score0.8/'

# def get_ckc_word(data_path,dataset):
ckc_files = [ckc_path + f for f in os.listdir(ckc_path)]  # 用idx.pkl中的idx排序
ckc_set = defaultdict(list)
for file in ckc_files:
    file_data = open(file,'r').readlines()
    for line in file_data:
        #createdby	Durex	countable	0.95331
        line = line.strip('\n').split('\t')
        head, tail = line[0], line[1]
        # if head not in ckc_set:
        if tail not in ckc_set[head]:
            ckc_set[head].append(tail)
    # file_data.c
    # return ckc_set

#TODO print(content)
# for content_id,content_val in two_set.items():
    # print(content_id)
    # print(content_set[str(content_id)])
#TODO print(con)


content_path = cwd+ '/Twitter_data/data/' + 'twitter15' + '/sig_concept.txt'

# content_set = defaultdict(list)

doc_lst =[]
for content_id,content_val in without_set.items():
    # print(content_id)
    for entity in doc_entity[str(content_id)]:

        if entity in ckc_set:
            # print(str(content_id)+',')
            doc_lst.append(content_id)
            # print(ckc_set[entity])
            break



for doc_id in doc_lst:
    print('----------------')
    # if len(doc_concept_entity)
    # doc_id = str(doc_id)
    # for entity in doc_entity[str(content_id)]:
    print(doc_id)
    print(content_set[str(doc_id)])
    # print(without_set[doc_id])

    # print(content_set)


    print(doc_concept_entity[str(doc_id)])
    # print(con_set[doc_id])

    doc_ckc = defaultdict(list)

    for entity in doc_entity[str(doc_id)]:
        # print(ckc_set[entity])
        if entity not in doc_ckc:
            doc_ckc[entity].append(ckc_set[entity])
    print(doc_ckc)
    # print(without_set[doc_id])








