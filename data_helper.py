import random
# from config import Config
import numpy as np
import pickle
# 配置文件
# conf = Config()
random.seed(99)
class Data_helper:
    def __init__(self):
        self.file_paths = ["./data/simtrain_to05sts_same.txt", "./data/simtrain_to05sts.txt"]
    # 产生字典
    def gen_vocab_dict(self, output_file = "./data/vocab.txt"):
        data = []
        words = []
        for path in self.file_paths:
            with open(path, encoding="utf8") as f:
                data.extend(f.readlines())
        for i in data:
            a = i.split("\t")[1]
            words.extend(a)
        with open(output_file, "w", encoding="utf8") as f2:
            words = ["[UNK]","[PAD]"]+list(set(words))
            for word in words:
                f2.write(word+"\n")
    # 切分train,dev,test集
    def gen_train_dev_test(self, data):
        return data[:int(0.9*len(data))], data[int(0.9*len(data)):int(0.95*len(data))], data[int(0.95*len(data)):len(data)]

     # 获取数据
    def gen_data_file(self, file_path="./data/simtrain_to05sts.txt"):
        data = []
        for path in self.file_paths:
            with open(path, encoding="utf8") as f:
                data.extend(f.readlines())

        random.shuffle(data)
        train, dev, test = self.gen_train_dev_test(data)
        with open("./data/train.txt", "w", encoding="utf8") as f:
            f.writelines(train)
        with open("./data/dev.txt", "w", encoding="utf8") as f:
            f.writelines(dev)
        with open("./data/test.txt", "w", encoding="utf8") as f:
            f.writelines(test)
            # return train, dev, test
    # 获取query, doc, label
    # def get_qdl(self, data):
    #     query = []
    #     doc = []
    #     label = []
    #     for i in data:
    #         query.append(i.split("\t")[1])
    #         doc.append(i.split("\t")[3])
    #         label.append(i.split("\t")[4].strip())
    #     return query, doc, label
    #
    # # 将sentence转为id
    # def sentense2id(self, sentense):
    #     with open("./data/words.txt", encoding="utf8") as f:
    #         word_map = {}
    #         sentense_token = []
    #         for line in f.readlines():
    #             word_map[line.replace("\n","").split("\t")[1]] = line.split("\t")[0]
    #         for word in sentense:
    #             if word not in word_map:
    #                 sentense_token.append(-1)
    #             else:
    #                 sentense_token.append(word_map[word])
    #
    #         # sentense 最长为25
    #         if len(sentense_token) > conf.max_sentense_len:
    #             sentense_token = sentense_token[conf.max_sentense_len]
    #         else:
    #             sentense_token = list(sentense_token + [0] * (conf.max_sentense_len - len(sentense_token)))
    #         return sentense_token
    # #创建train_tokens, dev_tokens


#
#
dh = Data_helper()
dh.gen_vocab_dict()
dh.gen_data_file()
# train, dev, test = dh.gen_data_file()
# train_query, train_doc, train_label = dh.get_qdl(train)
# dev_query, dev_doc, dev_label = dh.get_qdl(dev)
# test_query, test_doc, test_label = dh.get_qdl(test)
# # print(train_query, train_doc, train_label)
# train_query_tokens = list(map(dh.sentense2id, train_query))
# train_doc_tokens = list(map(dh.sentense2id, train_doc))
# dev_query_tokens = list(map(dh.sentense2id, dev_query))
# dev_doc_tokens = list(map(dh.sentense2id, dev_doc))
# with open("./data/train/train_query_tokens.pk", "wb") as f:
#     pickle.dump(train_query_tokens, f)
# with open("./data/train/train_doc_tokens.pk", "wb") as f:
#     pickle.dump(train_doc_tokens, f)
# with open("./data/train/train_label.pk", "wb") as f:
#     pickle.dump(train_label, f)
# with open("./data/dev/dev_query_tokens.pk", "wb") as f:
#     pickle.dump(dev_query_tokens, f)
# with open("./data/dev/dev_doc_tokens.pk", "wb") as f:
#     pickle.dump(dev_doc_tokens, f)
# with open("./data/dev/dev_label.pk", "wb") as f:
#     pickle.dump(dev_label, f)



