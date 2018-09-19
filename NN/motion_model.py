import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import csv
from itertools import combinations

class Motion(object):
    width = 48
    height = 48
    motions_ = ["b", "bw", "t", "p", "cc", "cf", "r", "w", "n"]
    motions =  ["batting", "batting_waiting", "throwing", "pitching", "catch_catcher", "catch_field", "run", "walking", "nope"]
    rgb = 3

    @staticmethod
    def conv2d(input, name, filter_size, output_channel, strides=[1, 1, 1, 1]):
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

    @staticmethod
    def deconv2d(input, name, filter_size, output_channel, strides=[1, 1]):
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

    @staticmethod
    def maxpool(input, name, filter_size, strides=[1, 2, 2, 1]):
        with tf.variable_scope(name):
            out = tf.nn.max_pool(value=input,
                                 ksize=[1, filter_size, filter_size, 1],
                                 strides=strides,
                                 padding="SAME")
        return out

    @staticmethod
    def upsample(input, name, times):
        with tf.variable_scope(name):
            out = tf.image.resize_bilinear(images=input,
                                           size=[int(input.shape[1] * times), int(input.shape[2] * times)])
        return out

    @staticmethod
    def fc(input, name, output_channel):
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
        self.epoch = 200
        self.lr = 0.001

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
            out = self.fc(out, name="fc_1", output_channel= 12 * 12 * 5)
            out = tf.nn.dropout(out, keep_prob=self.keep_prob)

            out = self.fc(out, name="fc_2", output_channel= 12 * 12 * 3)
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
        folder_name = "./_data/_motion/"

        dataset = []

        filenames = os.listdir(folder_name)
        for filename in filenames[::5]:
            full_filename = os.path.join(folder_name, filename)
            ext = os.path.splitext(full_filename)[-1]
            if ext == '.jpg':
                sett = {"X":cv2.resize(cv2.imread(full_filename), (self.width, self.height))}
                dataset.append(sett)

        print(len(dataset))
        return dataset

    def train(self):
        dataset = self.load_data()
        x = np.array([i["X"] for i in dataset])

        #batch_x = [j for j in [X[self.batch_size * i:self.batch_size * i + self.batch_size:1] for i in range(len(X))] if j != []]

        sess = tf.Session()

        optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        sess.run(tf.global_variables_initializer())
        all_vars = tf.global_variables()
        cae = [k for k in all_vars if k.name.startswith("CAE")]
        saver = tf.train.Saver(cae)
        saver.restore(sess, './_model/motion/CAE/CAE.ckpt')

        xs = []
        ys = []
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
            if (total_cost / total_batch < 1800):
                break

            xs.append(e + 1)
            ys.append(total_cost / total_batch)

        print("complete")
        saver.save(sess, './_model/motion/CAE/CAE.ckpt')
        plt.plot(xs, ys, 'b')
        plt.show()

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

class Classifier(Motion):
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

                if(i == 0):
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

    @staticmethod
    def last_relevant(seq, length):
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
                    X = [cv2.resize(cv2.imread(folder_name + str(i) + ".jpg"), (self.width, self.height)) for j in range(i, i+self.length) if(j <= int(line[1]))]
                    leng = len(X)

                    if (leng < self.length):
                        num_pad = self.length - leng
                        X = X + [pad for j in range(num_pad)]

                    dataset.append({"X" : X, "Y" : m, "L" : leng})
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

    def test(self, person_seq):
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
        if (max(score[0]) > 0.6):
            return self.motions[output[0]]
        else:
            return None


class TRN(Motion):
    def __init__(self, sess, istest=0):
        self.sess = sess

        self.batch_size = 100
        self.epoch = 200
        self.lr = 0.0005

        self.length = 12
        self.num_hidden = 64

        self.max_num_relation = 4

        self.image = tf.placeholder(tf.float32, [None, self.length, self.height, self.width, self.rgb])
        self.Y = tf.placeholder(tf.float32, [None, len(self.motions)])
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

        print(features)


        g = {}
        g_result = {}
        with tf.variable_scope("g_w_and_b", reuse=tf.AUTO_REUSE):
            for re in range(2, self.max_num_relation + 1):
                W = tf.get_variable(name='w_' + str(re),
                                    shape=[64 * re, 100],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                b = tf.get_variable(name='b_' + str(re),
                                    shape=[100],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))

                g[str(re)] = [W, b]
                g_result[str(re)] = tf.constant([0.0 for i in range(100)])

        h = {}
        h_result = tf.constant([0.0 for i in range(len(self.motions))])
        with tf.variable_scope("h_w_and_b", reuse=tf.AUTO_REUSE):
            for re in range(2, self.max_num_relation + 1):
                W = tf.get_variable(name='w_' + str(re),
                                    shape=[100, len(self.motions)],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                b = tf.get_variable(name='b_' + str(re),
                                    shape=[len(self.motions)],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))

                h[str(re)] = [W, b]

        for re in range(2, self.max_num_relation + 1):
            g_combi = [sorted(item) for item in combinations([i for i in range(self.length)], re)]
            for f in g_combi:
                g_conc = tf.concat([features[:, i, :] for i in f], 1)
                g_result[str(re)] = tf.add(g_result[str(re)],
                                           tf.nn.leaky_relu(tf.add(tf.matmul(g_conc, g[str(re)][0]), g[str(re)][1])))

        for re in range(2, self.max_num_relation + 1):
            h_result = tf.add(h_result,
                              tf.add(tf.matmul(g_result[str(re)], h[str(re)][0]), h[str(re)][1]))

        self.model = h_result
        print(self.model)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y))
        self.output = tf.nn.softmax(self.model)

        is_correct = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        if(istest == 1):
            all_vars = tf.global_variables()
            cls = [k for k in all_vars if k.name.startswith("CAE") or k.name.startswith("g_w_and_b") or k.name.startswith("h_w_and_b")]
            saver = tf.train.Saver(cls)
            saver.restore(self.sess, './_model/motion/trn/trn.ckpt')

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
                X = [cv2.resize(cv2.imread(folder_name + str(i) + ".jpg"), (self.width, self.height)) for i in range(start, end)]
                X_ = [X[i:i + self.length] for i in range(0, len(X), self.length)]

                for i in X_:
                    if(len(i) < self.length):
                        num_pad = self.length - len(i)
                        dataset.append({"X": i+[pad for _ in range(num_pad)], "Y": m})
                    else:
                        dataset.append({"X": i, "Y": m})

        return dataset

    def train(self):
        dataset = self.load_data()
        train_x = np.array([i["X"] for i in dataset])
        _y = np.array([i["Y"] for i in dataset])
        train_y = np.zeros((len(_y), len(self.motions)))
        train_y[np.arange(len(_y)), [i for i in _y]] = 1

        optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        self.sess.run(tf.global_variables_initializer())
        all_vars = tf.global_variables()

        cae = [k for k in all_vars if k.name.startswith("CAE")]
        saver = tf.train.Saver(cae)
        saver.restore(self.sess, './_model/motion/CAE/CAE.ckpt')

        cls = [k for k in all_vars if k.name.startswith("CAE") or k.name.startswith("g_w_and_b") or k.name.startswith("h_w_and_b")]
        saver2 = tf.train.Saver(cls)
        saver2.restore(self.sess, './_model/motion/trn/trn.ckpt')

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
                else:
                    batch_x = train_x[j:j + self.batch_size]
                    batch_y = train_y[j:j + self.batch_size]
                    j = j + self.batch_size

                _, cost_val, acc = self.sess.run([optimizer, self.cost, self.accuracy], feed_dict={self.image: batch_x, self.Y: batch_y, self.keep_prob: 0.75})

                total_cost = total_cost + cost_val

            print('Epoch:', '%d' % (e + 1), 'Average cost =', '{:.3f}'.format(total_cost / total_batch), ", Accuracy = ", acc)

            if (total_cost / total_batch < 0.3):
                break

            xs.append(e + 1)
            ys.append(total_cost / total_batch)

        print("complete")
        saver2.save(self.sess, './_model/motion/trn/trn.ckpt')
        plt.plot(xs, ys, 'b')
        plt.show()

    def test(self, person_seq):
        person_seq = person_seq[-self.length:]

        dataset = [cv2.resize(d, (self.width, self.height)) for d in person_seq]
        leng = len(dataset)

        pad = np.zeros((self.width, self.height, self.rgb))
        if (leng < self.length):
            num_pad = self.length - leng
            dataset = dataset + [pad for _ in range(num_pad)]

        dataset = np.array([dataset])


        score, output = self.sess.run([self.output, tf.argmax(self.output, 1)],
                                      feed_dict={self.image: dataset, self.keep_prob: 1})
        if (max(score[0]) > 0.9):
            return self.motions[output[0]]
        else:
            return None

"""
sess = tf.Session()
c = Classifier(sess, istest=0)
c.train()

cae = CAE()
cae.train()
"""

"""
sess = tf.Session()
c = TRN(sess)
c.train()
"""