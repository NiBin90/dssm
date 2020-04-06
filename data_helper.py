import random
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
    def gen_data_file(self):
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

dh = Data_helper()
dh.gen_vocab_dict()
dh.gen_data_file()
