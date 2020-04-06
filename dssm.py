from config import Config
import data_input
import numpy as np
import tensorflow as tf
import time
import random
conf = Config()

batch_size = 20
# batch size
L1_N = 300
L2_N = 128
# 获取数据的BOW（Bag of Word）输入
data_train = data_input.get_data_bow(conf.file_train)
data_dev = data_input.get_data_bow(conf.file_dev)
train_epoch_steps = int(len(data_train) / batch_size) - 1
dev_epoch_steps = int(len(data_dev) / batch_size) - 1


def add_fc_layer(inputs, n_input, n_output, activation=None):
    wlimit = np.sqrt(6.0 / (n_input + n_output))
    weights = tf.Variable(tf.random_uniform([n_input, n_output], -wlimit, wlimit))
    biases = tf.Variable(tf.random_uniform([n_output], -wlimit, wlimit))
    outputs = tf.matmul(inputs, weights)+biases
    if activation:
        outputs = tf.nn.relu(outputs)
    return outputs

def get_cosine_score(query_arr, doc_arr):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(query_arr), 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(doc_arr), 1))
    pooled_mul_12 = tf.reduce_sum(tf.multiply(query_arr, doc_arr), 1)
    cos_scores = pooled_mul_12/(pooled_len_1*pooled_len_2)
    return cos_scores

# define Iput layer
with tf.name_scope("Input"):
    query_batch = tf.placeholder(tf.float32, shape=[None, None], name="query_batch")
    doc_batch = tf.placeholder(tf.float32, shape=[None, None], name="doc_batch")
    doc_label_batch = tf.placeholder(tf.float32, shape=[None], name="doc_label_batch")
    keep_prob = tf.placeholder(tf.float32, name='drop_out_prob')
# define FC1 layer
with tf.name_scope("FC1"):
    query_layer1 = add_fc_layer(query_batch, conf.nwords, L1_N, activation=None)
    doc_layer1 = add_fc_layer(doc_batch, conf.nwords, L1_N, activation=None)

with tf.name_scope('Drop_out'):
    query_layer1 = tf.nn.dropout(query_layer1, keep_prob)
    doc_layer1 = tf.nn.dropout(doc_layer1, keep_prob)
# define FC2 layer
with tf.name_scope("FC2"):
    query_layer2 = add_fc_layer(query_layer1, L1_N, L2_N, activation=None)
    doc_layer2 = add_fc_layer(doc_layer1, L1_N, L2_N, activation=None)

with tf.name_scope("Cousin_Similarity"):
    consin_score = get_cosine_score(query_layer2, doc_layer2)
    cos_sim_prob = tf.clip_by_value(consin_score, 1e-8, 1.0)


with tf.name_scope("Loss"):
    cross_entropy = -tf.reduce_sum(doc_label_batch * tf.log(cos_sim_prob)+(1-doc_label_batch) * tf.log(1-cos_sim_prob))
    # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=doc_label_batch, logits=cos_sim_prob)
    losses = tf.reduce_mean(cross_entropy)
    tf.summary.scalar("loss", losses)

with tf.name_scope("Training"):
    train_step = tf.train.AdamOptimizer(conf.lr).minimize(losses)

with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        one = tf.ones_like(cos_sim_prob)
        zero = tf.zeros_like(cos_sim_prob)
        prediction = tf.where(cos_sim_prob <0.5, x=zero, y=one)
        correct_prediction = tf.equal(prediction, doc_label_batch)
    with tf.name_scope("acc"):
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", acc)

merged = tf.summary.merge_all()

with tf.name_scope('Test'):
    average_loss = tf.placeholder(tf.float32)
    loss_summary = tf.summary.scalar('average_loss', average_loss)

with tf.name_scope('Train'):
    train_average_loss = tf.placeholder(tf.float32)
    train_loss_summary = tf.summary.scalar('train_average_loss', train_average_loss)

def gen_batch(data_set, batch_id):
    data_batch = data_set[batch_id*batch_size: (batch_id+1)*batch_size]
    query_in = [x[0] for x in data_batch]
    doc_in = [x[1] for x in data_batch]
    label = [x[2] for x in data_batch]
    return query_in, doc_in, label

def feed_dict(data_set, batch_id, drop_prob):
    query_in, doc_in, label = gen_batch(data_set, batch_id)
    return {query_batch: query_in, doc_batch: doc_in, doc_label_batch: label, keep_prob: drop_prob}

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(conf.sumaries_dir+'/train', sess.graph)

    start = time.time()
    for epoch in range(conf.n_epoch):
        random.shuffle(data_train)
        for batch_id in range(train_epoch_steps):
            sess.run(train_step, feed_dict=feed_dict(data_train, batch_id, 0.5))

        end = time.time()
        # train loss
        epoch_loss, epoch_acc = 0, 0
        for i in range(train_epoch_steps):
            loss_t= sess.run(losses, feed_dict=feed_dict(data_train, i, 1))
            acc_t = sess.run(acc, feed_dict=feed_dict(data_train, i, 1))
            epoch_acc += acc_t
            epoch_loss += loss_t

        epoch_loss /= (train_epoch_steps)
        epoch_acc /=(train_epoch_steps)
        train_loss = sess.run(train_loss_summary, feed_dict={train_average_loss: epoch_loss})
        train_writer.add_summary(train_loss, epoch + 1)
        print("\nEpoch %d | Train Loss: %-4.3f | Train Acc: %-4.3f | SingleTrainTime: %3.3fs" %
            (epoch, epoch_loss, epoch_acc, end - start))
        # test loss
        start = time.time()
        epoch_loss, epoch_acc = 0, 0
        for i in range(dev_epoch_steps):
            loss_v = sess.run(losses, feed_dict=feed_dict(data_dev, i, 1))
            acc_v = sess.run(acc, feed_dict=feed_dict(data_dev, i, 1))
            epoch_loss += loss_v
            epoch_acc += acc_v
        epoch_loss /= (dev_epoch_steps)
        epoch_acc /= (dev_epoch_steps)
        test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
        train_writer.add_summary(test_loss, epoch + 1)
        # test_writer.add_summary(test_loss, step + 1)
        print("Epoch %d | Test  Loss: %-4.3f | Test  Acc: %-4.3f | SingleTestTime: %-3.3fs" %
              (epoch, epoch_loss, epoch_acc, start - end))
    # 保存模型
    save_path = saver.save(sess, "model/dssm.ckpt")
    print("Model saved in file: ", save_path)
