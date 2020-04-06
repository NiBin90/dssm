from config import Config
import numpy as np
conf = Config()
vocab_map = conf.vocab_map

def seq2bow(seq):
    bow_ids = np.zeros(conf.nwords)
    for w in seq:
        if w in vocab_map:
            bow_ids[vocab_map[w]] += 1
        else:
            bow_ids[vocab_map[conf.unk]] += 1
    return bow_ids

def get_data_bow(file_path):
    data_arr = []
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            _, query, __, doc, label = spline
            label = int(float(label))
            if label>2: label=1
            else: label = 0
            query_ids = seq2bow(query)
            doc_ids = seq2bow(doc)
            data_arr.append([query_ids, doc_ids, label])
    return data_arr
