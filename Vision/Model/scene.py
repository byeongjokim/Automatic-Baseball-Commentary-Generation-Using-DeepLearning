import cv2
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
from collections import Counter

class Scene_Model():
    def __init__(self, sess, istest=0):
        print("init scene_model")
        self.scenes = ["pitchingbatting", "batter", "closeup", "coach", "gallery", "frst", "center outfield", "right outfield", "second", "etc", "third", "left outfield", "ss"]

        self.width = 224
        self.height = 224
        self.num_label = 13
        self.ratio_crop = 1

        self.sess = sess
        self.batch_size = 30
        self.epoch = 200
        self.lr = 0.0005
        self.rgb = 3

        self.pre_train_ckpt = './_model/scene/pretrain/scene.ckpt'
        self.ckpt = './_model/scene/scene.ckpt'

        self.X = tf.placeholder(tf.float32, [None, self.width, self.height, self.rgb])
        self.Y = tf.placeholder(tf.float32, [None, self.num_label])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope("scene"):
            C1 = self.conv_layer(filter_size=3, fin=self.rgb, fout=64, din=self.X, name='C1')
            C1_2 = self.conv_layer(filter_size=3, fin=64, fout=64, din=C1, name='C1_2')
            P1 = self.pool(C1_2, option="maxpool")
            P1 = tf.nn.dropout(P1, keep_prob=self.keep_prob)

            C2 = self.conv_layer(filter_size=3, fin=64, fout=128, din=P1, name='C2')
            #C2_2 = self.conv_layer(filter_size=3, fin=128, fout=128, din=C2, name='C2_2')
            P2 = self.pool(C2, option="maxpool")
            P2 = tf.nn.dropout(P2, keep_prob=self.keep_prob)

            C3_1 = self.conv_layer(filter_size=3, fin=128, fout=256, din=P2, name='C3_1')
            C3_2 = self.conv_layer(filter_size=3, fin=256, fout=256, din=C3_1, name='C3_2')
            P3 = self.pool(C3_2, option="maxpool")
            P3 = tf.nn.dropout(P3, keep_prob=self.keep_prob)

            C4_1 = self.conv_layer(filter_size=3, fin=256, fout=512, din=P3, name='C4_1')
            C4_2 = self.conv_layer(filter_size=3, fin=512, fout=512, din=C4_1, name='C4_2')
            P4 = self.pool(C4_2, option="maxpool")
            P4 = tf.nn.dropout(P4, keep_prob=self.keep_prob)

            C5_1 = self.conv_layer(filter_size=3, fin=512, fout=512, din=P4, name='C5_1')
            C5_2 = self.conv_layer(filter_size=3, fin=512, fout=512, din=C5_1, name='C5_2')
            self.P5 = self.pool(C5_2, option="maxpool")
            P5 = tf.nn.dropout(self.P5, keep_prob=self.keep_prob)

            fc0 = tf.reshape(P5, [-1, 7 * 7 * 512])

            with tf.device("/cpu:0"):
                W1 = tf.get_variable("W1", shape=[7 * 7 * 512, 4096], initializer=tf.contrib.layers.xavier_initializer())
                b1 = tf.get_variable("b1", shape=[4096], initializer=tf.contrib.layers.xavier_initializer())
                fc1 = tf.nn.relu(tf.matmul(fc0, W1) + b1)

                W2 = tf.get_variable("W2", shape=[4096, 1000], initializer=tf.contrib.layers.xavier_initializer())
                b2 = tf.get_variable("b2", shape=[1000], initializer=tf.contrib.layers.xavier_initializer())
                fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)

                W3 = tf.get_variable("W3", shape=[1000, self.num_label], initializer=tf.contrib.layers.xavier_initializer())
                b3 = tf.get_variable("b3", shape=[self.num_label], initializer=tf.contrib.layers.xavier_initializer())
                self.model = tf.matmul(fc2, W3) + b3

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y))
        self.output = tf.nn.softmax(self.model)

        if(istest==1):
            all_vars = tf.global_variables()
            scene = [k for k in all_vars if k.name.startswith("scene")]
            saver = tf.train.Saver(scene)
            saver.restore(self.sess, self.ckpt)

    def conv_layer(self, filter_size, fin, fout, din, name):
        with tf.variable_scope(name):
            W = tf.get_variable(name=name + "_W", shape=[filter_size, filter_size, fin, fout],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name=name + "_b", shape=[fout], initializer=tf.contrib.layers.xavier_initializer(0.0))
            C = tf.nn.conv2d(din, W, strides=[1, 1, 1, 1], padding='SAME')
            R = tf.nn.relu(tf.nn.bias_add(C, b))
            return R

    def pool(self, din, option='maxpool'):
        if (option == 'maxpool'):
            pool = tf.nn.max_pool(din, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        elif (option == 'avrpool'):
            pool = tf.nn.avg_pool(din, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        else:
            return din
        return pool

    def crop_img(self, img, scale=1.0):
        center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
        width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
        left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
        top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
        img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
        return img_cropped

    def load_data(self, play):
        data_set = []

        for p in play:
            print(p)
            folder_path = "_data/" + p + "/"
            csv_path = folder_path + p + ".csv"

            dataset = []
            f = open(csv_path, "r")
            reader = csv.reader(f)
            for line in reader:
                if(int(line[0]) < int(line[1]) and int(line[1]) - int(line[0]) < 200):
                    sett = {"start":line[0], "end":line[1], "label":line[2]}
                    if (sett["label"] == "13"):  #right field
                        sett["label"] = "9"
                    dataset.append(sett)
            f.close()

            for i in dataset:
                if(i["label"] == "0" or i["label"] == "1" or i["label"] == "2"):
                    interval = 3
                else:
                    interval = 1

                for j in range(int(i["start"]), int(i["end"])+1, interval):
                    if(self.rgb == 1):
                        image = cv2.resize(
                            self.crop_img(
                                cv2.cvtColor(
                                    cv2.imread(folder_path + str(j) + ".jpg"),
                                    cv2.COLOR_BGR2GRAY
                                ), self.ratio_crop),
                            (self.width, self.height)
                        )
                    elif(self.rgb == 3):
                        image = cv2.resize(
                            self.crop_img(
                                cv2.imread(folder_path + str(j) + ".jpg"),
                                self.ratio_crop),
                            (self.width, self.height)
                        )

                    data_set.append({"image":image, "label":int(i["label"])})

                    if (int(i["label"]) == 5):  #first base
                        image = cv2.flip(image, 1)
                        data_set.append({"image":image, "label":10})
                    if (int(i["label"]) == 10):  # third base
                        image = cv2.flip(image, 1)
                        data_set.append({"image": image, "label": 5})

                    if (int(i["label"]) == 11):  # left field
                        image = cv2.flip(image, 1)
                        data_set.append({"image": image, "label": 7})
                    if (int(i["label"]) == 7):  #right field
                        image = cv2.flip(image, 1)
                        data_set.append({"image":image, "label":11})


        X = np.array([i["image"] for i in data_set])
        _y = np.array([i["label"] for i in data_set])
        Y = np.zeros((len(_y), len(set(_y))))
        Y[np.arange(len(_y)), [i for i in _y]] = 1

        num_data = len(X)
        print(num_data)
        """
        num_test = 30
        num_test = num_test * -1

        num_validation = 100
        num_validation = num_validation * -1
        
        self.train_x = X[:num_validation + num_test]
        self.train_y = Y[:num_validation + num_test]

        self.valid_x = X[num_validation + num_test:num_test]
        self.valid_y = Y[num_validation + num_test:num_test]

        self.test_x = X[num_test:].reshape(-1, self.width, self.height, self.rgb)
        self.test_y = Y[num_test:]
        
        print(str(num_data) + " data, " + str(len(self.train_x)) + " train data " + str(len(self.valid_x)) + " valid data " + str(len(self.test_x)) + " test data ")
        """
        self.train_x = X
        self.train_y = Y

    def train(self):
        optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        self.sess.run(tf.global_variables_initializer())
        all_vars = tf.global_variables()

        scene = [k for k in all_vars if k.name.startswith("scene")]
        saver = tf.train.Saver(scene)
        saver.restore(self.sess, self.pre_train_ckpt)

        xs = []
        ys = []

        total_batch = int(len(self.train_x) / self.batch_size)
        for e in range(self.epoch):
            total_cost = 0

            j = 0
            for i in range(total_batch):
                if (j + self.batch_size > len(self.train_x)):
                    batch_x = self.train_x[j:]
                    batch_y = self.train_y[j:]
                else:
                    batch_x = self.train_x[j:j + self.batch_size]
                    batch_y = self.train_y[j:j + self.batch_size]
                    j = j + self.batch_size

                batch_x = batch_x.reshape(-1, self.width, self.height, self.rgb)
                batch_y = batch_y.reshape(-1, self.num_label)

                _, cost_val = self.sess.run([optimizer, self.cost], feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.75})

                total_cost = total_cost + cost_val

            print('Epoch:', '%d' % (e + 1), 'Average cost =', '{:.3f}'.format(total_cost / total_batch))

            if (total_cost / total_batch < 0.03):
                break

            xs.append(e+1)
            ys.append(total_cost / total_batch)


        print("complete")
        saver.save(self.sess, self.ckpt)
        plt.plot(xs, ys, 'b')
        plt.show()

    def get_accuracy(self, v):
        folder_path = "../kbo/_data/" + v + "/"
        text_path = folder_path + v + ".txt"

        dataset = []
        f = open(text_path, "r")

        dataset = []
        while True:
            line = f.readline()
            if not line: break
            line = line.split(", ")


            sett = {"label": int(line[0]), "image_num": line[1].rstrip() + ".jpg"}
            dataset.append(sett)

        data_set = []
        for d in dataset:
            image = cv2.resize(
                self.crop_img(
                    cv2.imread(folder_path + d["image_num"]),
                    self.ratio_crop),
                (self.width, self.height)
            )
            data_set.append({"image":image, "label":d["label"]})

        X = np.array([i["image"] for i in data_set if(i["label"] == 12 or i["label"] == 11)])
        _y = np.array([i["label"] for i in data_set if(i["label"] == 12 or i["label"] == 11)])
        print(Counter(_y))
        print(len(set(_y)))
        Y = np.zeros((len(_y), 13))
        Y[np.arange(len(_y)), [i-1 for i in _y]] = 1

        is_correct = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        batch_size = 10

        total_batch = int(len(X) / batch_size)

        acc = 0.0

        j = 0
        for i in range(total_batch):
            if (j + batch_size > len(X)):
                batch_x = X[j:]
                batch_y = Y[j:]
            else:
                batch_x = X[j:j + batch_size]
                batch_y = Y[j:j + batch_size]
                j = j + batch_size

            acc = acc + self.sess.run(accuracy, feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 1}) * 100

        print(acc/total_batch)

    def predict(self, image):
        image = cv2.resize(image, (self.width, self.height))
        if(self.rgb == 1):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(
            self.crop_img(image, self.ratio_crop),
            (self.width, self.height)
        )

        image_X = image.reshape(-1, self.width, self.height, self.rgb)

        score, output, P5 = self.sess.run([self.output, tf.argmax(self.output, 1), self.P5],
                                      feed_dict={self.X: image_X, self.keep_prob: 1})

        return output[0], max(score[0]), np.reshape(P5, (7, 7, 512))