import tensorflow as tf
from tensorflow.contrib import rnn
import cv2
from os import walk
import os
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from NN.cnn import conv_layer, pool
import math
import csv

class Motion():
    width = 48
    height = 48
    #motions = ["batting", "catching", "running", "standing", "throwing", "walking"]
    #motions = ["batting", "catching", "running", "throwing", "walking"]
    motions = ["t", "b", "w", "r", "c", "n"]
    motions_full = ["throwing", "batting", "walking", "running", "catching", "nope"]

    def conv2d(self, input, name, kshape, strides=[1, 1, 1, 1]):
        with tf.variable_scope(name):
            W = tf.get_variable(name='w_' + name,
                                shape=kshape,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable(name='b_' + name,
                                shape=[kshape[3]],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            out = tf.nn.conv2d(input, W, strides=strides, padding='SAME')
            out = tf.nn.bias_add(out, b)
            out = tf.nn.leaky_relu(out)
            return out

    def deconv2d(self, input, name, kshape, n_outputs, strides=[1, 1]):
        with tf.variable_scope(name):
            out = tf.contrib.layers.conv2d_transpose(input,
                                                     num_outputs=n_outputs,
                                                     kernel_size=kshape,
                                                     stride=strides,
                                                     padding='SAME',
                                                     weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(
                                                         uniform=False),
                                                     biases_initializer=tf.contrib.layers.xavier_initializer(
                                                         uniform=False),
                                                     activation_fn=tf.nn.leaky_relu)
            return out

    def maxpool2d(self, x, name, kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
        with tf.variable_scope(name):
            out = tf.nn.max_pool(x,
                                 ksize=kshape,  # size of window
                                 strides=strides,
                                 padding='SAME')
            return out

    def upsample(self, input, name, factor=[2, 2]):
        size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
        with tf.variable_scope(name):
            out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
            return out

    def fullyConnected(self, input, name, output_size):
        with tf.variable_scope(name):
            input_size = input.shape[1:]
            input_size = int(np.prod(input_size))  # get total num of cells in one input image
            W = tf.get_variable(name='w_' + name,
                                shape=[input_size, output_size],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable(name='b_' + name,
                                shape=[output_size],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            input = tf.reshape(input, [-1, input_size])
            out = tf.nn.leaky_relu(tf.add(tf.matmul(input, W), b))
            return out

    def dropout(self, input, name, keep_rate):
        with tf.variable_scope(name):
            out = tf.nn.dropout(input, keep_rate)
            return out

    def leak(self, x):
        p = 0.2
        with tf.variable_scope("leak"):
            f1 = 0.5 * (1 + p)
            f2 = 0.5 * (1 - p)
            return f1 * x + f2 * abs(x)

class CAE(Motion):
    def __init__(self, sess):
        print("motion")
        self.sess = sess
        self.batch_size = 100
        self.epoch = 200
        self.lr = 0.001

    def load_data(self):
        folder_name = "_data/_motion/"

        dataset = []
        sett = {"X":None}

        filenames = os.listdir(folder_name)
        for filename in filenames:
            full_filename = os.path.join(folder_name, filename)
            ext = os.path.splitext(full_filename)[-1]
            if ext == '.jpg':
                sett = {"X":

                                    cv2.resize(
                                              cv2.imread(full_filename),
                                              (self.width, self.height))
                }
                dataset.append(sett)
        return dataset

    def autoencoder(self):

        input = tf.placeholder(tf.float32, [None, self.width, self.height, 3])

        with tf.variable_scope("motion"):

            c1 = self.conv2d(input, name='c1', kshape=[7, 7, 3, 15])  # Input: [48,48,3];  Output: [48,48,15]
            p1 = self.maxpool2d(c1, name='p1')  # Input: [48,48,15]; Output: [24,24,15]
            do1 = self.dropout(p1, name='do1', keep_rate=0.75)
            c2 = self.conv2d(do1, name='c2', kshape=[5, 5, 15, 25])  # Input: [24,24,15]; Output: [24,24,25]
            p2 = self.maxpool2d(c2, name='p2')  # Input: [24,24,25]; Output: [12,12,25]
            p2 = tf.reshape(p2, shape=[-1, 12 * 12 * 25])  # Input: [12,12,25]; Output: [12*12*25]
            fc1 = self.fullyConnected(p2, name='fc1', output_size=12 * 12 * 5)  # Input: [12*12*25]; Output: [12*12*5]
            do2 = self.dropout(fc1, name='do2', keep_rate=0.75)
            fc2 = self.fullyConnected(do2, name='fc2', output_size=12 * 12 * 3)  # Input: [12*12*5];  Output: [12*12*3]
            do3 = self.dropout(fc2, name='do3', keep_rate=0.75)
            fc3 = self.fullyConnected(do3, name='fc3',
                                 output_size=64)  # Input: [12*12*3];  Output: [64] --> bottleneck layer
            # Decoding part
            fc4 = self.fullyConnected(fc3, name='fc4', output_size=12 * 12 * 3)  # Input: [64];       Output: [12*12*3]
            do4 = self.dropout(fc4, name='do4', keep_rate=0.75)
            fc5 = self.fullyConnected(do4, name='fc5', output_size=12 * 12 * 5)  # Input: [12*12*3];  Output: [12*12*5]
            do5 = self.dropout(fc5, name='do5', keep_rate=0.75)
            fc6 = self.fullyConnected(do5, name='fc6', output_size=21 * 21 * 25)  # Input: [12*12*5];  Output: [12*12*25]
            do6 = self.dropout(fc6, name='do6', keep_rate=0.75)
            do6 = tf.reshape(do6, shape=[-1, 21, 21, 25])  # Input: [12*12*25]; Output: [12,12,25]
            dc1 = self.deconv2d(do6, name='dc1', kshape=[5, 5], n_outputs=15)  # Input: [12,12,25]; Output: [12,12,15]
            up1 = self.upsample(dc1, name='up1', factor=[2, 2])  # Input: [12,12,15]; Output: [24,24,15]
            dc2 = self.deconv2d(up1, name='dc2', kshape=[7, 7], n_outputs=3)  # Input: [24,24,15]; Output: [24,24,3]
            up2 = self.upsample(dc2, name='up2', factor=[2, 2])  # Input: [24,24,3];  Output: [48,48,3]
            output = self.fullyConnected(up2, name='output', output_size=48 * 48 * 3)

        cost = tf.reduce_mean(tf.square(tf.subtract(output, tf.reshape(input, shape=[-1, 48 * 48 * 3]))))
        return {"input":input, "output":tf.reshape(output, shape=[-1, 48, 48, 3]), "cost":cost}

    def train(self):
        dataset = self.load_data()
        print(len(dataset))
        shuffle(dataset)
        shuffle(dataset)

        ae = self.autoencoder()

        optimizer = tf.train.AdamOptimizer(self.lr).minimize(ae["cost"])
        self.sess.run(tf.global_variables_initializer())

        all_vars = tf.global_variables()
        print(all_vars)
        motion = [k for k in all_vars if k.name.startswith("motion")]
        print(motion)
        self.saver = tf.train.Saver(motion)

        #self.saver.restore(self.sess, './_model/motion/motion.ckpt')

        x = np.array([i["X"] for i in dataset])

        xs = []
        ys = []

        total_batch = int(len(dataset) / self.batch_size)

        if (total_batch == 0):
            total_batch = 1

        for e in range(self.epoch):
            total_cost = 0

            j = 0
            for i in range(total_batch):
                if (j + self.batch_size > len(x)):
                    batch_x = x[j:]

                else:
                    batch_x = x[j:j + self.batch_size]
                    j = j + self.batch_size

                #batch_x = batch_x.reshape(-1, self.width, self.height, 3)

                _, cost_val = self.sess.run([optimizer, ae["cost"]],
                                            feed_dict={ae["input"]: batch_x})

                total_cost = total_cost + cost_val

            print('Epoch:', '%d' % (e + 1), 'Average cost =', '{:.3f}'.format(total_cost / total_batch))

            if (total_cost / total_batch < 1700):
                break

            xs.append(e + 1)
            ys.append(total_cost / total_batch)

        print("complete")
        self.saver.save(self.sess, './_model/motion/motion.ckpt')
        plt.plot(xs, ys, 'b')
        plt.show()

    def test(self):
        img = "9375.jpg"
        image = "_data/_motion/" + img
        image = np.resize(cv2.resize(cv2.imread(image), (self.width, self.height)), (1, self.width, self.height, 3))

        ae = self.autoencoder()

        all_vars = tf.global_variables()
        print(all_vars)
        motion = [k for k in all_vars if k.name.startswith("motion")]
        print(motion)
        self.saver = tf.train.Saver(motion)
        self.saver.restore(self.sess, './_model/motion/motion.ckpt')
        o = self.sess.run(ae["output"], feed_dict={ae["input"]: image})
        o = np.resize(o, [self.width, self.height, 3])
        print(self.sess.run(tf.reduce_sum(o)))
        cv2.imwrite(img, o)

class Classifier(Motion):
    def __init__(self, sess, istest=1):
        print("motion")
        self.sess = sess
        self.length = 3

        self.num_hidden = 64
        self.batch_size = 20
        self.epoch = 200
        self.lr = 0.0005

        self.model()
        if(istest == 1):
            all_vars = tf.global_variables()
            cls = [k for k in all_vars if k.name.startswith("motion") or k.name.startswith("lstm")]
            self.saver = tf.train.Saver(cls)
            self.saver.restore(self.sess, './_model/motion/cls/motion.ckpt')
            print(cls)

    def load_data(self):
        folder_name = "_data/_motion/"
        csv_path = folder_name + "motion.csv"

        f = open(csv_path, "r")
        reader = csv.reader(f)
        pad = np.zeros((self.width, self.height, 3))

        dataset = []
        for line in reader:
            sett = {"X": None, "Y": None}

            if(int(line[0]) < int(line[1])):
                sett = {"start": line[0], "end":line[1], "label":line[2]}
                l = int(line[1]) - int(line[0]) + 1

                for i in range(int(line[0]), int(line[1])):
                    X = []
                    for j in range(i, i+self.length):
                        if(j > int(line[1])):
                            break
                        img = cv2.resize(
                                    cv2.imread(folder_name + str(i) + ".jpg"),
                                    (self.width, self.height)
                                )
                        X.append(img)

                    if (len(X) < self.length):
                        num_pad = self.length - len(X)
                        for i in range(num_pad):
                            X.append(pad)
                    dataset.append({"X" : X, "Y" : self.motions.index(line[2])})

        return dataset


    def model(self):
        self.input = tf.placeholder(tf.float32, [None, self.length, self.width, self.height, 3])
        self.Y = tf.placeholder(tf.float32, [None, len(self.motions)])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope("motion", reuse=tf.AUTO_REUSE):
            for i in range(self.length):
                img = self.input[:, i, :, :, :]
                c1 = self.conv2d(img, name='c1', kshape=[7, 7, 3, 15])
                p1 = self.maxpool2d(c1, name='p1')
                do1 = self.dropout(p1, name='do1', keep_rate=self.keep_prob)
                c2 = self.conv2d(do1, name='c2', kshape=[5, 5, 15, 25])
                p2 = self.maxpool2d(c2, name='p2')
                p2 = tf.reshape(p2, shape=[-1, 12 * 12 * 25])
                fc1 = self.fullyConnected(p2, name='fc1', output_size=12 * 12 * 5)
                do2 = self.dropout(fc1, name='do2', keep_rate=self.keep_prob)
                fc2 = self.fullyConnected(do2, name='fc2', output_size=12 * 12 * 3)
                do3 = self.dropout(fc2, name='do3', keep_rate=self.keep_prob)
                fc3 = self.fullyConnected(do3, name='fc3', output_size=64)
                re = tf.reshape(fc3, shape=[-1, 1, 64])

                if (i == 0):
                    result = re
                else:
                    result = tf.concat([result, re], 1)

        print(result)

        with tf.variable_scope("lstm"):
            x = tf.unstack(result, self.length, 1)
            lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
            outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

            W = tf.Variable(tf.random_normal([self.num_hidden, len(self.motions)]))
            b = tf.Variable(tf.random_normal([len(self.motions)]))

        self.model = tf.matmul(outputs[-1], W) + b

        print(self.model)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y))

        self.softmax = tf.nn.softmax(self.model)

    def train(self):
        dataset = self.load_data()
        shuffle(dataset)

        optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        train_x = np.array([i["X"] for i in dataset])
        _y = np.array([i["Y"] for i in dataset])
        train_y = np.zeros((len(_y), len(self.motions)))
        train_y[np.arange(len(_y)), [i for i in _y]] = 1
        print(train_x.shape)
        print(train_y.shape)

        self.sess.run(tf.global_variables_initializer())
        all_vars = tf.global_variables()
        print(all_vars)
        motion = [k for k in all_vars if k.name.startswith("motion")]
        print(motion)
        saver = tf.train.Saver(motion)
        saver.restore(self.sess, './_model/motion/motion.ckpt')

        cls = [k for k in all_vars if k.name.startswith("motion") or k.name.startswith("lstm")]
        print(cls)

        self.saver2 = tf.train.Saver(cls)

        xs = []
        ys = []

        total_batch = int(len(dataset) / self.batch_size)

        if (total_batch == 0):
            total_batch = 1

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

                #batch_x = batch_x.reshape(-1, self.length, self.width, self.height, 3)
                #batch_y = batch_y.reshape(-1, len(self.motions))

                _, cost_val = self.sess.run([optimizer, self.cost],
                                            feed_dict={self.input: batch_x, self.Y: batch_y,
                                                       self.keep_prob: 0.8})

                total_cost = total_cost + cost_val

            print('Epoch:', '%d' % (e + 1), 'Average cost =', '{:.3f}'.format(total_cost / total_batch))

            if (total_cost / total_batch < 0.05):
                break

            xs.append(e + 1)
            ys.append(total_cost / total_batch)

        print("complete")
        self.saver2.save(self.sess, './_model/motion/cls/motion.ckpt')
        plt.plot(xs, ys, 'b')
        plt.show()

    def test(self, person):
        person = person[-self.length:]

        dataset = []
        for d in person:
            d = cv2.resize(d, (self.width, self.height))
            #cv2.imshow("test", d)
            #cv2.waitKey(0)
            dataset.append(d)

        pad = np.zeros((self.width, self.height, 3))
        if (len(dataset) < self.length):
            num_pad = self.length - len(dataset)
            for _ in range(num_pad):
                dataset.append(pad)

        dataset = np.array([dataset])
        output = self.sess.run(tf.argmax(self.softmax, 1),
                                    feed_dict={self.input: dataset,
                                               self.keep_prob: 1})
        #print(self.motions_full[output[0]])
        return self.motions_full[output[0]]

    def draw_box(self, frame, person_bbox):
        index = ["A", "B", "C", "D", "E"]

        for i in index:
            if person_bbox[i]:
                bbox = person_bbox[i][-1]
                frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                frame = cv2.putText(frame, person_bbox[i + "_"], (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2)

        return frame

    def insert_person(self, frame, person_bbox, person_image, bbox):
        index = ["A", "B", "C", "D", "E"]

        for i in index:
            p = frame[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]
            if not (person_bbox[i]):
                person_bbox[i].append(bbox)
                person_image[i].append(p)
                break

            else:
                iou = self.cal_mean_iou(bbox, [person_bbox[i][-1]])
                if (0.5 < iou):
                    person_bbox[i].append(bbox)
                    person_image[i].append(p)

                    person_bbox[i + "_"] = self.test(person_image[i])
                    break
                else:
                    pass

        return person_bbox, person_image

    def cal_mean_iou(self, bbox1, bboxes2):
        s = 0.0
        for bbox2 in bboxes2:
            min_x = max(bbox1[0], bbox2[0])
            max_x = min(bbox1[2], bbox2[2])
            min_y = max(bbox1[1], bbox2[1])
            max_y = min(bbox1[3], bbox2[3])

            if (max_x < min_x or max_y < min_y):
                s = s + 0.0
            else:
                inter = (max_x - min_x) * (max_y - min_y)
                bb1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                bb2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

                iou = inter / (bb1 + bb2 - inter)
                if (iou >= 0.0 and iou <= 1.0):
                    s = s + iou
                else:
                    s = s + 0.0
        return s / len(bboxes2)
