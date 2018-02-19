import cv2
import os
import csv
import tensorflow as tf
import numpy as np
from imutils.object_detection import non_max_suppression
from NN.cnn import conv_layer, pool

class Motion_model():

    width = 224
    height = 224

    max_num = 10

    kind_motion = ["pitching", "swinging", "running"]

    def __init__(self):
        print("init motion model")

    def get_human(self, image):
        body_cascade = cv2.CascadeClassifier('./_data/cascades/haarcascade_fullbody.xml')
        upper_body_cascade = cv2.CascadeClassifier('./_data/cascades/haarcascade_upperbody.xml')

        people = body_cascade.detectMultiScale(image, 1.05, 3, flags=cv2.CASCADE_SCALE_IMAGE)
        people = non_max_suppression(people, probs=None, overlapThresh=0.75)

        if len(people) == 0:
            people = upper_body_cascade.detectMultiScale(image, 1.05, 3, flags=cv2.CASCADE_SCALE_IMAGE)
            people = non_max_suppression(people, probs=None, overlapThresh=0.75)
            full = 0
        else:
            full = 1
        return people, full

    def load_data(self):
        csv_path = "./_data/motion_data.csv"
        folder_path = "./motion_data/train/"

        dataset=[]
        f = open(csv_path, "r")
        reader = csv.reader(f)
        for line in reader:
            sett = {"start":line[0], "end":line[1], "label":line[2], "interval":line[3]}

            dataset.append(sett)
        f.close()


        self.cnn_X = []
        __cnn_y = []
        train_data = []
        for i in dataset:
            image_set = []
            for j in range(int(i["start"]), int(i["end"])+1):
                image = cv2.resize(
                    cv2.cvtColor(
                        cv2.imread(folder_path + str(j) + ".jpg"),
                        cv2.COLOR_BGR2GRAY
                    ),
                    (self.width, self.height)
                )

                image_set.append(image)

                self.cnn_X.append(image)
                __cnn_y.append(i["label"])

            remain = self.max_num - len(image_set)
            image_set = image_set + [np.zeros((self.width, self.height)) for i in range(remain)]
            image_set = np.array(image_set)

            sett = {"image" : image_set, "label" : i["label"]}
            train_data.append(sett)

        self.cnn_X = np.array(self.cnn_X)
        _cnn_y = np.array([i for i in __cnn_y])
        cnn_y = np.zeros((len(_cnn_y), len(set(_cnn_y))))
        cnn_y[np.arange(len(_cnn_y)), [self.kind_motion.index(i) for i in _cnn_y]] = 1
        self.cnn_Y = cnn_y


        self.rnn_X = np.array([i["image"] for i in train_data])
        _rnn_y = np.array([i["label"] for i in train_data])
        rnn_y = np.zeros((len(_rnn_y), len(set(_rnn_y))))
        rnn_y[np.arange(len(_rnn_y)), [self.kind_motion.index(i) for i in _rnn_y]] = 1
        self.rnn_Y = rnn_y

        self.num_label = len(set(_rnn_y))

        print(self.cnn_X.shape)
        print(self.cnn_Y.shape)
        print(self.rnn_X.shape)
        print(self.rnn_Y.shape)

        return 1

    def CNN_pretrain(self):
        self.cnn_X = tf.placeholder(tf.float32, [None, self.width, self.height, 1])
        self.cnn_Y = tf.placeholder(tf.float32, [None, self.num_label])
        self.cnn_keep_prob = tf.placeholder(tf.float32)

        C1_1 = conv_layer(filter_size=3, fin=1, fout=64, din=self.cnn_X, name='model_C1_1')
        C1_2 = conv_layer(filter_size=3, fin=64, fout=64, din=C1_1, name='model_C1_2')
        P1 = pool(C1_2, option="maxpool")

        C2_1 = conv_layer(filter_size=3, fin=64, fout=128, din=P1, name='model_C2_1')
        C2_2 = conv_layer(filter_size=3, fin=128, fout=128, din=C2_1, name='model_C2_2')
        P2 = pool(C2_2, option="maxpool")

        C3_1 = conv_layer(filter_size=3, fin=128, fout=256, din=P2, name='model_C3_1')
        C3_2 = conv_layer(filter_size=3, fin=256, fout=256, din=C3_1, name='model_C3_2')
        C3_3 = conv_layer(filter_size=1, fin=256, fout=256, din=C3_2, name='model_C3_3')
        P3 = pool(C3_3, option="maxpool")

        C4_1 = conv_layer(filter_size=3, fin=256, fout=512, din=P3, name='model_C4_1')
        C4_2 = conv_layer(filter_size=3, fin=512, fout=512, din=C4_1, name='model_C4_2')
        C4_3 = conv_layer(filter_size=1, fin=512, fout=512, din=C4_2, name='model_C4_3')
        P4 = pool(C4_3, option="maxpool")

        C5_1 = conv_layer(filter_size=3, fin=512, fout=512, din=P4, name='model_C5_1')
        C5_2 = conv_layer(filter_size=3, fin=512, fout=512, din=C5_1, name='model_C5_2')
        C5_3 = conv_layer(filter_size=1, fin=512, fout=512, din=C5_2, name='model_C5_3')
        P5 = pool(C5_3, option="maxpool")

        print(P5)

        fc0 = tf.reshape(P5, [-1, 7 * 7 * 512])

        with tf.device("/cpu:0"):
            W1 = tf.get_variable("model_W1", shape=[7 * 7 * 512, 4096], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([4096]))
            fc1 = tf.nn.relu(tf.matmul(fc0, W1) + b1)

            W2 = tf.get_variable("model_W2", shape=[4096, 4096], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([4096]))
            fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)




    def make_model(self):
        X = tf.placeholder(tf.float32, [None, None, self.width, self.height])
        Y = tf.placeholder(tf.float32, [None, self.num_label])
        self.keep_prob = tf.placeholder(tf.float32)




        return 1

    def train(self):
        return 1

    def test(self):
        return 1

    def predict(self, image):
        return 1