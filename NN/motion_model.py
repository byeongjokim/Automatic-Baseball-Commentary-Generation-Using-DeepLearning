import tensorflow as tf
import cv2
from NN.tinyYOLOv2.test import ObjectDetect
from os import walk
import os
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

class Motion():
    def __init__(self, sess):
        print("motion")
        self.no = 0
        self.width = 224
        self.height = 224

        self.sess = sess
        self.batch_size = 30
        self.epoch = 300

    def save_data(self):

        self.o = ObjectDetect(self.sess)

        folder_name = "_data/_motion/"
        motions = ["run", "walk", "handwave"]

        for m in motions:
            files = []
            for (dirpath, dirnames, filenames) in walk(folder_name + m + "/video/"):
                files.extend(filenames)

            for v in files:
                self.save_video2person(m, folder_name + m + "/video/" + v)

            self.no = 0


    def save_video2person(self, m, v):
        #video = "_data/_motion/run/video/person01_running_d1_uncomp.avi"
        path = "_data/_motion/" + m
        cap = cv2.VideoCapture(v)

        while(True):
            ret, frame = cap.read()

            if not ret:
                break

            if self.no > 3000:
                break

            h, w, c = frame.shape
            ratio_h = h / 416
            ratio_w = w / 416

            result = self.o.predict(frame)
            if result:
                for obj in result:
                    if(obj[2] == 'person'):
                        cv2.imwrite(path + "/" + m + str(self.no) + ".jpg", frame[int(obj[0][1]*ratio_h) : int(obj[0][3]*ratio_h), int(obj[0][0]*ratio_w) : int(obj[0][2]*ratio_w)])
                        self.no = self.no + 1

    def load_data(self):
        folder_name = "_data/_motion/"
        motions = ["run", "walk", "handwave"]

        dataset = []
        for m in motions:
            sett = {"image":None, "label":None}

            filenames = os.listdir(folder_name + m)
            for filename in filenames:
                full_filename = os.path.join(folder_name + m, filename)
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.jpg':
                    #print(full_filename)
                    sett = {"image": cv2.cvtColor(
                                                cv2.resize(
                                                          cv2.imread(full_filename),
                                                          (self.width, self.height)),
                                                cv2.COLOR_BGR2GRAY),
                            "label": motions.index(m)}
                    dataset.append(sett)
        return dataset

    def model(self):
        self.X = tf.placeholder(tf.float32, [None, self.width, self.height, 1])
        self.Y = tf.placeholder(tf.float32, [None, 3])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope("motion"):
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 6], stddev=0.01))
            C1 = tf.nn.conv2d(self.X, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(C1)
            P1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            D1 = tf.nn.dropout(P1, keep_prob=self.keep_prob)

            W2 = tf.Variable(tf.random_normal([3, 3, 6, 12], stddev=0.01))
            C2 = tf.nn.conv2d(D1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(C2)
            P2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            D2 = tf.nn.dropout(P2, keep_prob=self.keep_prob)

            W3 = tf.Variable(tf.random_normal([3, 3, 12, 12], stddev=0.01))
            C3 = tf.nn.conv2d(D2, W3, strides=[1, 1, 1, 1], padding='SAME')
            L3 = tf.nn.relu(C3)
            P3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            D3 = tf.nn.dropout(P3, keep_prob=self.keep_prob)

            print(D3)

            D3 = tf.reshape(D3, [-1, 28*28*12])

            #W4 = tf.get_variable("W4", shape=[7 * 13 * 12, 100],
            #                     initializer=tf.contrib.layers.xavier_initializer())
            W4 = tf.Variable(tf.random_normal([28 * 28 * 12, 100], stddev=0.01))
            print(W4)
            b4 = tf.Variable(tf.random_normal([100]))
            L4 = tf.nn.relu(tf.matmul(D3, W4) + b4)
            D4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

            W5 = tf.Variable(tf.random_normal([100, 3], stddev=0.01))
            b5 = tf.Variable(tf.random_normal([3]))
            self.model = tf.matmul(D4, W5) + b5

        print(self.model)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

        self.softmax = tf.nn.softmax(self.model)

        all_vars = tf.global_variables()
        motion = [k for k in all_vars if k.name.startswith("motion")]

        self.saver = tf.train.Saver(motion)
        self.saver.restore(self.sess, './_model/motion/motion.ckpt')

    def train(self):
        dataset = self.load_data()
        shuffle(dataset)
        shuffle(dataset)

        self.sess.run(tf.global_variables_initializer())

        train_x = np.array([i["image"] for i in dataset])
        _y = np.array([i["label"] for i in dataset])
        train_y = np.zeros((len(_y), len(set(_y))))
        train_y[np.arange(len(_y)), [i for i in _y]] = 1

        train_x = train_x[:5000]
        train_y = train_y[:5000]

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

                batch_x = batch_x.reshape(-1, self.width, self.height, 1)
                batch_y = batch_y.reshape(-1, 3)

                _, cost_val = self.sess.run([self.optimizer, self.cost],
                                            feed_dict={self.X: batch_x, self.Y: batch_y,
                                                       self.keep_prob: 0.8})

                total_cost = total_cost + cost_val

            print('Epoch:', '%d' % (e + 1), 'Average cost =', '{:.3f}'.format(total_cost / total_batch))

            if (total_cost / total_batch < 0.03):
                break

            xs.append(e + 1)
            ys.append(total_cost / total_batch)

        print("complete")
        self.saver.save(self.sess, './_model/motion/motion.ckpt')
        plt.plot(xs, ys, 'b')
        plt.show()

    def test(self, image):
        image = cv2.resize(image, (self.width, self.height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x = image.reshape(-1, self.width, self.height, 1)


        result = self.sess.run(tf.argmax(self.softmax, 1), feed_dict={self.X: x, self.keep_prob: 1})

        print(result)
        result = self.sess.run(self.softmax, feed_dict={self.X: x, self.keep_prob: 1})
        print(result)