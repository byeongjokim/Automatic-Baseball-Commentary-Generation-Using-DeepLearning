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

    cnn_batch_size = 2
    cnn_chk = './_model/motion/cnn/cnn.ckpt'
    cnn_ckpt = tf.train.get_checkpoint_state(("./_model/motion/cnn/"))

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


        _cnn_X = []
        _rnn_X = []
        __cnn_y = []
        __rnn_y = []

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

                _cnn_X.append(image)
                __cnn_y.append(i["label"])

            remain = self.max_num - len(image_set)
            image_set = image_set + [np.zeros((self.width, self.height)) for i in range(remain)]
            image_set = np.array(image_set)

            _rnn_X.append(image_set)
            __rnn_y.append(i["label"])

        self.cnn_X = np.array(_cnn_X)
        _cnn_y = np.array(__cnn_y)
        cnn_y = np.zeros((len(_cnn_y), len(set(_cnn_y))))
        cnn_y[np.arange(len(_cnn_y)), [self.kind_motion.index(i) for i in _cnn_y]] = 1
        self.cnn_Y = cnn_y


        self.rnn_X = np.array(_rnn_X)
        _rnn_y = np.array(__rnn_y)
        rnn_y = np.zeros((len(_rnn_y), len(set(_rnn_y))))
        rnn_y[np.arange(len(_rnn_y)), [self.kind_motion.index(i) for i in _rnn_y]] = 1
        self.rnn_Y = rnn_y

        self.num_label = len(set(_rnn_y))

        print(self.cnn_X.shape)
        print(self.cnn_Y.shape)
        print(self.rnn_X.shape)
        print(self.rnn_Y.shape)

    def CNN_pretrain(self):
        self.CNN_X = tf.placeholder(tf.float32, [None, self.width, self.height, 1])
        self.CNN_Y = tf.placeholder(tf.float32, [None, self.num_label])
        self.cnn_keep_prob = tf.placeholder(tf.float32)

        C1_1 = conv_layer(filter_size=3, fin=1, fout=64, din=self.CNN_X, name='model_C1_1')
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

            W3 = tf.get_variable("model_W3", shape=[4096, self.num_label], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([self.num_label]))
            self.cnn_model = tf.matmul(fc2, W3) + b3


        self.cnn_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cnn_model, labels=self.CNN_Y))
        self.cnn_optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cnn_cost)

        self.cnn_sess = tf.Session()
        self.cnn_saver = tf.train.Saver()

        if self.cnn_ckpt and tf.train.checkpoint_exists(self.cnn_ckpt.model_checkpoint_path):
            print("rstore the sess!!")
            self.cnn_saver.restore(self.cnn_sess, self.cnn_chk)
        else:
            self.cnn_sess.run(tf.global_variables_initializer())

    def CNN_train(self):
        print("start cnn pre train")
        train_x = self.cnn_X
        train_y = self.cnn_Y

        total_batch = int(47 / self.cnn_batch_size)

        if (total_batch == 0):
            total_batch = 1

        for e in range(1000):
            total_cost = 0

            j = 0
            for i in range(total_batch):
                if (j + self.cnn_batch_size > 47):
                    batch_x = train_x[j:]
                    batch_y = train_y[j:]
                else:
                    batch_x = train_x[j:j + self.cnn_batch_size]
                    batch_y = train_y[j:j + self.cnn_batch_size]
                    j = j + self.cnn_batch_size

                batch_x = batch_x.reshape(-1, self.width, self.height, 1)

                _, cost_val = self.cnn_sess.run([self.cnn_optimizer, self.cnn_cost], feed_dict={self.CNN_X: batch_x, self.CNN_Y: batch_y})

                total_cost = total_cost + cost_val

            if (total_cost / total_batch < 0.03):
                break
            print('Epoch:', '%d' % (e + 1), 'Average cost =', '{:.3f}'.format(total_cost / total_batch))

        print("complete")
        self.cnn_saver.save(self.cnn_sess, self.cnn_chk)

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