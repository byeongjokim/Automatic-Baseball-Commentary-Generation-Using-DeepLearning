import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import csv
from itertools import combinations
import collections
import random

class Motion(object):
    width = 48
    height = 48
    motions_ = ["b", "bw", "t", "p", "cc", "cf", "r", "w", "n"]
    motions =  ["batting", "batting_waiting", "throwing", "pitching", "catch_catcher", "catch_field", "run", "walking", "nope"]
    rgb = 3

    def conv2d(self, input, name, filter_size, output_channel, strides=[1, 1, 1, 1]):
        input_channel = input.shape[3]
        with tf.variable_scope(name):
            W = tf.get_variable(name=name+"_W",
                                shape=[filter_size, filter_size, input_channel, output_channel],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable(name=name+"_b",
                                shape=[output_channel],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            c = tf.nn.conv2d(input=input,
                             filter=W,
                             strides=strides,
                             padding="SAME")
            c = tf.nn.bias_add(c, b)
            c = tf.nn.leaky_relu(c)
            return c

    def deconv2d(self, input, name, filter_size, output_channel, strides=[1, 1]):
        with tf.variable_scope(name):
            out = tf.layers.conv2d_transpose(inputs=input,
                                       filters=output_channel,
                                       kernel_size=[filter_size, filter_size],
                                       strides=strides,
                                       padding="SAME",
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       activation=tf.nn.leaky_relu)
        return out

    def maxpool(self, input, name, filter_size, strides=[1, 2, 2, 1]):
        with tf.variable_scope(name):
            out = tf.nn.max_pool(value=input,
                                 ksize=[1, filter_size, filter_size, 1],
                                 strides=strides,
                                 padding="SAME")
        return out

    def upsample(self, input, name, times):
        with tf.variable_scope(name):
            out = tf.image.resize_bilinear(images=input,
                                           size=[int(input.shape[1] * times), int(input.shape[2] * times)])
        return out

    def fc(self, input, name, output_channel):
        input_channel = input.shape[1]
        with tf.variable_scope(name):
            W = tf.get_variable(name=name+"_W",
                                shape=[input_channel, output_channel],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable(name=name+"_b",
                                shape=[output_channel],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            out = tf.nn.leaky_relu(tf.add(tf.matmul(input, W), b))
        return out

class CAE(Motion):
    def __init__(self):
        self.batch_size = 100
        self.epoch = 300
        self.lr = 0.0001

        self.image = tf.placeholder(tf.float32, [None, self.height, self.width, self.rgb])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope("CAE"):
            out = self.conv2d(self.image, name="conv_1", filter_size=7, output_channel=15)
            out = self.maxpool(out, name="pool_1", filter_size=2)
            out = tf.nn.dropout(out, keep_prob=self.keep_prob)

            out = self.conv2d(out, name="conv_2", filter_size=5, output_channel=25)
            out = self.maxpool(out, name="pool_2", filter_size=2)
            out = tf.nn.dropout(out, keep_prob=self.keep_prob)

            out = tf.reshape(out, shape=[-1, 12 * 12 * 25])
            out = self.fc(out, name="fc_1", output_channel=12 * 12 * 5)
            out = tf.nn.dropout(out, keep_prob=self.keep_prob)

            out = self.fc(out, name="fc_2", output_channel=12 * 12 * 3)
            out = tf.nn.dropout(out, keep_prob=self.keep_prob)

            out = self.fc(out, name="fc_3", output_channel=64)

            out = self.fc(out, name="fc_4", output_channel=12 * 12 * 3)
            out = tf.nn.dropout(out, keep_prob=self.keep_prob)

            out = self.fc(out, name="fc_5", output_channel=12 * 12 * 5)
            out = tf.nn.dropout(out, keep_prob=self.keep_prob)

            out = self.fc(out, name="fc_6", output_channel=12 * 12 * 25)
            out = tf.nn.dropout(out, keep_prob=self.keep_prob)

            out = tf.reshape(out, shape=[-1, 12, 12, 25])

            out = self.deconv2d(out, name="dconv_1", filter_size=5, output_channel=15)
            out = self.upsample(out, name="ups_1", times=2)

            out = self.deconv2d(out, name="dconv_2", filter_size=7, output_channel=3)
            out = self.upsample(out, name="ups_2", times=2)

            out = tf.reshape(out, shape=[-1, 48 * 48 * 3])
            out = self.fc(out, name="fc_7", output_channel=48 * 48 * 3)

        self.output = tf.reshape(out, shape=[-1, 48, 48, 3])
        self.cost = tf.reduce_mean(tf.square(tf.subtract(out, tf.reshape(self.image, shape=[-1, 48 * 48 * 3]))))

    def load_data(self):
        folder_name = "./_data/motion/"

        dataset = []

        filenames = os.listdir(folder_name)

        for filename in filenames[::3]:
            full_filename = os.path.join(folder_name, filename)
            ext = os.path.splitext(full_filename)[-1]
            if ext == '.jpg':
                sett = {"X":cv2.resize(cv2.imread(full_filename), (self.width, self.height))}
                dataset.append(sett)

        return dataset

    def train(self):
        dataset = self.load_data()
        x = np.array([i["X"] for i in dataset])

        sess = tf.Session()

        optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        sess.run(tf.global_variables_initializer())
        all_vars = tf.global_variables()
        cae = [k for k in all_vars if k.name.startswith("CAE")]
        saver = tf.train.Saver(cae)
        #saver.restore(sess, './_model/motion/CAE/CAE.ckpt')

        total_batch = int(len(dataset) / self.batch_size)

        for e in range(self.epoch):
            total_cost = 0

            j = 0
            for i in range(total_batch):
                if (j + self.batch_size > len(x)):
                    batch_x = x[j:]
                else:
                    batch_x = x[j:j + self.batch_size]
                    j = j + self.batch_size

                _, cost_val = sess.run([optimizer, self.cost], feed_dict={self.image : batch_x, self.keep_prob : 0.75})
                total_cost = total_cost + cost_val

            print('Epoch:', '%d' % (e + 1), 'Average cost =', '{:.3f}'.format(total_cost / total_batch))

            if (total_cost / total_batch < 950):
                break

        print("complete")
        saver.save(sess, './_model/motion/CAE/CAE.ckpt')

        return 1

    def test(self):
        x = cv2.resize(cv2.imread("_data/_motion/1058.jpg"), (self.width, self.height))
        x = np.array([x])

        sess = tf.Session()
        all_vars = tf.global_variables()
        cae = [k for k in all_vars if k.name.startswith("CAE")]
        saver = tf.train.Saver(cae)
        saver.restore(sess, './_model/motion/CAE/CAE.ckpt')

        o = sess.run(self.output, feed_dict={self.image: x, self.keep_prob: 1})
        o = np.resize(o, [self.width, self.height, 3])

        cv2.imwrite('1a.jpg', o)

class Motion_Model(Motion):
    def __init__(self, sess, istest=0):
        self.sess = sess
        self.length = 10
        self.num_hidden = 64
        self.batch_size = 100
        self.epoch = 200
        self.lr = 0.0005

        self.image = tf.placeholder(tf.float32, [None, self.length, self.height, self.width, self.rgb])
        self.Y = tf.placeholder(tf.float32, [None, len(self.motions)])
        self.L = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope("CAE", reuse=tf.AUTO_REUSE):
            for i in range(self.length):
                input = self.image[:, i, :, :, :]
                out = self.conv2d(input, name="conv_1", filter_size=7, output_channel=15)
                out = self.maxpool(out, name="pool_1", filter_size=2)
                out = tf.nn.dropout(out, keep_prob=self.keep_prob)

                out = self.conv2d(out, name="conv_2", filter_size=5, output_channel=25)
                out = self.maxpool(out, name="pool_2", filter_size=2)
                out = tf.nn.dropout(out, keep_prob=self.keep_prob)

                out = tf.reshape(out, shape=[-1, 12 * 12 * 25])
                out = self.fc(out, name="fc_1", output_channel=12 * 12 * 5)
                out = tf.nn.dropout(out, keep_prob=self.keep_prob)

                out = self.fc(out, name="fc_2", output_channel=12 * 12 * 3)
                out = tf.nn.dropout(out, keep_prob=self.keep_prob)

                out = self.fc(out, name="fc_3", output_channel=64)
                out = tf.reshape(out, shape=[-1, 1, 64])

                if (i == 0):
                    features = out
                else:
                    features = tf.concat([features, out], 1)

        with tf.variable_scope("cls"):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_hidden)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            all_outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=features, sequence_length=self.L, dtype=tf.float32)
            outputs = self.last_relevant(all_outputs, self.L)

            W = tf.Variable(tf.random_normal([self.num_hidden, len(self.motions)]))
            b = tf.Variable(tf.random_normal([len(self.motions)]))

        self.model = tf.matmul(outputs, W) + b
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y))
        self.output = tf.nn.softmax(self.model)

        if(istest==1):
            all_vars = tf.global_variables()
            cls = [k for k in all_vars if k.name.startswith("CAE") or k.name.startswith("cls")]
            saver = tf.train.Saver(cls)
            saver.restore(self.sess, './_model/motion/cls/cls.ckpt')

    def last_relevant(self, seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)

    def load_data(self):
        folder_name = "./_data/_motion/"
        csv_path = folder_name + "motion.csv"

        f = open(csv_path, "r")
        reader = csv.reader(f)
        pad = np.zeros((self.width, self.height, self.rgb))

        dataset = []
        for line in reader:
            start = int(line[0])
            end = int(line[1])
            m = self.motions_.index(line[2])

            if(start < end):
                for i in range(start, end):
                    X = [cv2.resize(cv2.imread(folder_name + str(i) + ".jpg"), (self.width, self.height)) for j in range(i, i+self.length, 2) if(j <= int(line[1]))]
                    leng = len(X)

                    if (leng < self.length):
                        num_pad = self.length - leng
                        X = X + [pad for j in range(num_pad)]

                    dataset.append({"X" : X, "Y" : m, "L" : leng})

                    X = [cv2.resize(cv2.imread(folder_name + str(i) + ".jpg"), (self.width, self.height)) for j in range(i, i + self.length, 3) if (j <= int(line[1]))]
                    leng = len(X)

                    if (leng < self.length):
                        num_pad = self.length - leng
                        X = X + [pad for j in range(num_pad)]

                    dataset.append({"X": X, "Y": m, "L": leng})

                    X = [cv2.resize(cv2.imread(folder_name + str(i) + ".jpg"), (self.width, self.height)) for j in range(i, i + self.length, 4) if (j <= int(line[1]))]
                    leng = len(X)

                    if (leng < self.length):
                        num_pad = self.length - leng
                        X = X + [pad for j in range(num_pad)]

                    dataset.append({"X": X, "Y": m, "L": leng})


        return dataset

    def train(self):
        dataset = self.load_data()
        train_x = np.array([i["X"] for i in dataset])
        _y = np.array([i["Y"] for i in dataset])
        train_y = np.zeros((len(_y), len(self.motions)))
        train_y[np.arange(len(_y)), [i for i in _y]] = 1
        train_l = np.array([i["L"] for i in dataset])

        optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        self.sess.run(tf.global_variables_initializer())
        all_vars = tf.global_variables()

        cae = [k for k in all_vars if k.name.startswith("CAE")]
        saver = tf.train.Saver(cae)
        saver.restore(self.sess, './_model/motion/CAE/CAE.ckpt')

        cls = [k for k in all_vars if k.name.startswith("CAE") or k.name.startswith("cls")]
        saver2 = tf.train.Saver(cls)
        #saver2.restore(self.sess, './_model/motion/cls/cls.ckpt')

        xs = []
        ys = []

        total_batch = int(len(dataset) / self.batch_size)

        for e in range(self.epoch):
            total_cost = 0

            j = 0
            for i in range(total_batch):
                if (j + self.batch_size > len(train_x)):
                    batch_x = train_x[j:]
                    batch_y = train_y[j:]
                    batch_l = train_l[j:]
                else:
                    batch_x = train_x[j:j + self.batch_size]
                    batch_y = train_y[j:j + self.batch_size]
                    batch_l = train_l[j:j + self.batch_size]
                    j = j + self.batch_size

                _, cost_val = self.sess.run([optimizer, self.cost], feed_dict={self.image: batch_x, self.Y: batch_y, self.L: batch_l, self.keep_prob: 0.75})

                total_cost = total_cost + cost_val

            print('Epoch:', '%d' % (e + 1), 'Average cost =', '{:.3f}'.format(total_cost / total_batch))

            if (total_cost / total_batch < 0.1):
                break

            xs.append(e + 1)
            ys.append(total_cost / total_batch)

        print("complete")
        saver2.save(self.sess, './_model/motion/cls/cls.ckpt')
        plt.plot(xs, ys, 'b')
        plt.show()

    def predict(self, person_seq):
        person_seq = person_seq[-self.length:]

        dataset = [cv2.resize(d, (self.width, self.height)) for d in person_seq]
        leng = len(dataset)

        pad = np.zeros((self.width, self.height, self.rgb))
        if (leng < self.length):
            num_pad = self.length - leng
            dataset = dataset + [pad for j in range(num_pad)]

        dataset = np.array([dataset])
        leng = np.array([leng])

        score, output = self.sess.run([self.output, tf.argmax(self.output, 1)], feed_dict={self.image: dataset, self.L: leng, self.keep_prob: 1})
        if(max(score[0]) > 0.9):
            return output[0], max(score[0])
        else:
            return None, None

    def evaluation(self):
        score = {}
        motions = ["batting", "batting_waiting", "throwing", "pitching", "catch_catcher", "catch_field", "run", "walking", "nope"]
        for m in motions:
            score[m] = [0.0, 0.0]

        dataset = self.load_data()
        random.shuffle(dataset)
        random.shuffle(dataset)
        random.shuffle(dataset)
        random.shuffle(dataset)
        dataset = dataset[:500]
        print("fin dataset")

        Y = [d["Y"] for d in dataset]
        print(collections.Counter(Y))

        for d in dataset:
            x = [d["X"]]
            y = d["Y"]
            l = [d["L"]]
            output = self.sess.run(tf.argmax(self.output, 1), feed_dict={self.image: x, self.L: l, self.keep_prob: 1})

            motion = motions[output[0]]

            score[motion][1] = score[motion][1] + 1
            if(y == output[0]):
                score[motion][0] = score[motion][0] + 1

        mAP = 0
        for m in motions:
            mAP = mAP + score[m][0]/score[m][1]

        print(mAP/len(motions))
        print(score)

