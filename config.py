def load_vocab(file_path):
    word_dict = {}
    with open(file_path, encoding='utf8') as f:
        for idx, word in enumerate(f.readlines()):
            word = word.strip("\n")
            word_dict[word] = idx
    return word_dict

class Config:
    def __init__(self):
        self.vocab_map = load_vocab(self.vocab_path)
        self.nwords = len(self.vocab_map)
    max_sentense_len = 25
    vocab_path = "./data/vocab.txt"
    file_train = "./data/train.txt"
    file_dev = "./data/dev.txt"
    unk = "[UNK]"
    pad = "[PAD]"
    lr = 0.00005
    n_epoch = 20
    sumaries_dir = './summary/'
