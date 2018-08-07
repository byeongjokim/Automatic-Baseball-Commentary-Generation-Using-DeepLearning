import cv2
import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import csv
from NN.cnn import conv_layer, pool
import pytesseract as tes
from collections import Counter

class Scene_Model():
    chk_scene = './_model/scene/scene.ckpt'
    ckpt = tf.train.get_checkpoint_state(("./_model/scene"))

    batch_size = 30
    epoch = 1

    width = 224
    height = 224

    #kind_scene = ['field', 'pitcher', 'gallery', 'batter', 'pitchingbatting', 'beforestart', '1', 'coach', 'closeup', '3']
    #kind_scene = ["pitchingbatting", "1", "batter", "closeup", "coach", "gallery", "field", "etc", "3"]

    ratio_crop = 1

    num_label = 13
    rgb = 1

    def __init__(self, sess):
        print("init scene_model")
        self.sess = sess


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
            folder_path = "./_data/" + p + "/"
            csv_path = folder_path + p + ".csv"
            print(csv_path)

            dataset = []
            f = open(csv_path, "r")
            reader = csv.reader(f)
            for line in reader:
                if(int(line[0]) < int(line[1]) and int(line[1]) - int(line[0]) < 200):
                    sett = {"start":line[0], "end":line[1], "label":line[2]}
                    if (int(sett["label"]) != 11 and int(sett["label"]) != 12):
                        dataset.append(sett)

            f.close()

            interval = 4
            for i in dataset:
                #print(i["start"])
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

                    if (int(i["label"]) == 6):  #first base
                        image = cv2.flip(image, 1)
                        data_set.append({"image":image, "label":11})

                    if (int(i["label"]) == 8):  #right field
                        image = cv2.flip(image, 1)
                        data_set.append({"image":image, "label":12})

        random.shuffle(data_set)

        X = np.array([i["image"] for i in data_set])
        _y = np.array([i["label"] for i in data_set])
        print(Counter(_y))
        Y = np.zeros((len(_y), len(set(_y))))
        Y[np.arange(len(_y)), [i-1 for i in _y]] = 1

        num_data = len(X)

        num_test = 30
        num_test = num_test * -1

        num_validation = 100
        num_validation = num_validation * -1

        self.train_x = X[:num_validation + num_test]
        self.train_y = Y[:num_validation + num_test]

        self.valid_x = X[num_validation + num_test:num_test].reshape(-1, self.width, self.height, self.rgb)
        self.valid_y = Y[num_validation + num_test:num_test]

        self.test_x = X[num_test:].reshape(-1, self.width, self.height, self.rgb)
        self.test_y = Y[num_test:]

        print(str(num_data) + " data, " + str(len(self.train_x)) + " train data " + str(len(self.valid_x)) + " valid data " + str(len(self.test_x)) + " test data ")


    #vggnet - A
    def make_model(self):

        self.scene_X = tf.placeholder(tf.float32, [None, self.width, self.height, self.rgb])
        self.scene_Y = tf.placeholder(tf.float32, [None, self.num_label])
        self.scene_keep_prob = tf.placeholder(tf.float32)


        C1 = conv_layer(filter_size=3, fin=self.rgb, fout=64, din=self.scene_X, name='scene_C1')
        C1_2 = conv_layer(filter_size=3, fin=64, fout=64, din=C1, name='scene_C1_2')
        P1 = pool(C1_2, option="maxpool")
        P1 = tf.nn.dropout(P1, keep_prob=self.scene_keep_prob)

        C2 = conv_layer(filter_size=3, fin=64, fout=128, din=P1, name='scene_C2')
        C2_2 = conv_layer(filter_size=3, fin=128, fout=128, din=C2, name='scene_C2_2')
        P2 = pool(C2, option="maxpool")
        P2 = tf.nn.dropout(P2, keep_prob=self.scene_keep_prob)

        C3_1 = conv_layer(filter_size=3, fin=128, fout=256, din=P2, name='scene_C3_1')
        C3_2 = conv_layer(filter_size=3, fin=256, fout=256, din=C3_1, name='scene_C3_2')
        P3 = pool(C3_2, option="maxpool")
        P3 = tf.nn.dropout(P3, keep_prob=self.scene_keep_prob)

        C4_1 = conv_layer(filter_size=3, fin=256, fout=512, din=P3, name='scene_C4_1')
        C4_2 = conv_layer(filter_size=3, fin=512, fout=512, din=C4_1, name='scene_C4_2')
        P4 = pool(C4_2, option="maxpool")
        P4 = tf.nn.dropout(P4, keep_prob=self.scene_keep_prob)

        C5_1 = conv_layer(filter_size=3, fin=512, fout=512, din=P4, name='scene_C5_1')
        C5_2 = conv_layer(filter_size=3, fin=512, fout=512, din=C5_1, name='scene_C5_2')
        P5 = pool(C5_2, option="maxpool")
        P5 = tf.nn.dropout(P5, keep_prob=self.scene_keep_prob)

        print(P5)

        fc0 = tf.reshape(P5, [-1, 7 * 7 * 512])

        with tf.device("/cpu:0"):
            W1 = tf.get_variable("scene_W1", shape=[7 * 7 * 512, 4096],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([4096]))
            fc1 = tf.nn.relu(tf.matmul(fc0, W1) + b1)

            W2 = tf.get_variable("scene_W2", shape=[4096, 1000], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([1000]))
            fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)

            W3 = tf.get_variable("scene_W3", shape=[1000, self.num_label],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([self.num_label]))
            self.scene_model = tf.matmul(fc2, W3) + b3

        #"""

        print(self.scene_model)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scene_model, labels=self.scene_Y))
        self.optimizer = tf.train.AdamOptimizer(0.0005).minimize(self.cost)

        #self.sess = tf.Session()

        all_vars = tf.global_variables()
        scene = [k for k in all_vars if not (k.name.startswith("object") or k.name.startswith("motion") or k.name.startswith("lstm"))]
        print(scene)
        self.saver = tf.train.Saver(scene)

        self.sotfmax = tf.nn.softmax(self.scene_model)

        is_correct = tf.equal(tf.argmax(self.scene_model, 1), tf.argmax(self.scene_Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


        self.saver.restore(self.sess, self.chk_scene)




    def train(self):

        xs = []
        ys = []
        yv = []

        self.sess.run(tf.global_variables_initializer())

        total_batch = int(len(self.train_x) / self.batch_size)

        if (total_batch == 0):
            total_batch = 1
        validation_acc = self.sess.run(self.accuracy, feed_dict={self.scene_X: self.valid_x, self.scene_Y: self.valid_y, self.scene_keep_prob: 1}) * 100
        print("Validation Set Accuracy : ", validation_acc)
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

                _, cost_val = self.sess.run([self.optimizer, self.cost],
                                            feed_dict={self.scene_X: batch_x, self.scene_Y: batch_y, self.scene_keep_prob: 0.8})

                total_cost = total_cost + cost_val

            print('Epoch:', '%d' % (e + 1), 'Average cost =', '{:.3f}'.format(total_cost / total_batch))

            validation_acc = self.sess.run(self.accuracy, feed_dict={self.scene_X: self.valid_x, self.scene_Y: self.valid_y, self.scene_keep_prob: 1}) * 100
            print("Validation Set Accuracy : ", validation_acc)

            if (int(validation_acc) > 93):
                break

            if (total_cost / total_batch < 0.03):
                break


            xs.append(e+1)
            ys.append(total_cost / total_batch)

            yv.append(validation_acc)

        print("complete")
        self.saver.save(self.sess, self.chk_scene)
        plt.plot(xs, ys, 'b')
        plt.plot(xs, yv, 'r')
        plt.show()


    def test(self):

        is_correct = tf.equal(tf.argmax(self.scene_model, 1), tf.argmax(self.scene_Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        print('accuracy: ',
              self.sess.run(accuracy, feed_dict={self.scene_X: self.test_x, self.scene_Y: self.test_y, self.scene_keep_prob: 1}) * 100)

        print("Label: ", self.sess.run(tf.argmax(self.test_y, 1)))
        print("Prediction: ", self.sess.run(tf.argmax(self.scene_model, 1), feed_dict={self.scene_X: self.test_x, self.scene_keep_prob: 1}))

    def predict(self, image):
        image = cv2.resize(image, (self.width, self.height))
        if(self.rgb == 1):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(
            self.crop_img(image, self.ratio_crop),
            (self.width, self.height)
        )

        image_X = image.reshape(-1, self.width, self.height, self.rgb)

        '''
                if(result[0] == 3):
                    tes.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
                    results = tes.image_to_string(image)
                    if(results):
                        print(results +" 번 선수가 보이네요.")
                '''
        result, softmax = self.sess.run([tf.argmax(self.scene_model, 1), self.sotfmax], feed_dict={self.scene_X: image_X, self.scene_keep_prob: 1})

        score = max(softmax[0])

        return result[0], score
