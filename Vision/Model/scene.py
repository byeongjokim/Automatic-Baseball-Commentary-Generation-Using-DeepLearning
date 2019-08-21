import cv2
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
from collections import Counter

class Scene_Model():
    def __init__(self, sess):
        self.scenes = ["pitchingbatting", "batter", "closeup", "coach", "gallery", "frst", "center outfield", "right outfield", "second", "etc", "third", "left outfield", "ss"]

        self.width = 224
        self.height = 224
        self.rgb = 3
        self.num_label = 13
        self.ratio_crop = 1

        self.sess = sess

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


            W1 = tf.get_variable("W1", shape=[7 * 7 * 512, 4096], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable("b1", shape=[4096], initializer=tf.contrib.layers.xavier_initializer())
            fc1 = tf.nn.relu(tf.matmul(fc0, W1) + b1)

            W2 = tf.get_variable("W2", shape=[4096, 1000], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable("b2", shape=[1000], initializer=tf.contrib.layers.xavier_initializer())
            fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)

            W3 = tf.get_variable("W3", shape=[1000, self.num_label], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable("b3", shape=[self.num_label], initializer=tf.contrib.layers.xavier_initializer())
            self.model = tf.matmul(fc2, W3) + b3

        self.output = tf.nn.softmax(self.model)

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