# -*- coding: utf-8 -*-
import csv
import cv2
import os
import numpy as np
import tensorflow as tf
import operator

from sklearn.neighbors import NearestNeighbors
from skimage.measure import compare_ssim as ssim
from imutils.object_detection import non_max_suppression
from operator import itemgetter
from collections import Counter

from NN.cnn import conv_layer, pool
from NN.scene_model import Scene_Model


class SceneData():
    def __init__(self, Resources, shape=(320, 180)):
        print("init_sceneData")
        self.Resources = Resources

        self.width = shape[0]
        self.height = shape[1]

        self.scene = Scene_Model()
        self.scene.num_label = 10
        self.scene.make_model()


    def load_image_data(self):
        path = "./_data/scene_image/"
        image = []
        for (p, dir, files) in os.walk(path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.jpg':
                    image.append(filename)

        self.image_data = []
        image_data_X = []
        image_data_y = []
        for i in image:
            self.image_data.append({"image":cv2.resize(
                cv2.cvtColor(
                    cv2.imread(path+i),
                    cv2.COLOR_BGR2GRAY),
                (self.width, self.height)),
                "label":i.split(".")[0].split("_")[0]})

            image_data_X.append(cv2.resize(
                                    cv2.cvtColor(
                                        cv2.imread(path+i),
                                        cv2.COLOR_BGR2GRAY),
                                    (self.width, self.height)))
            image_data_y.append(i.split(".")[0].split("_")[0])

        print("we have %d image data" %(len(self.image_data)))


    def Annotation_with_frame(self, frame, relayText):
        #print("\t\t\t\t대기시간이 길어 영상처리로 텍스트 생성")

        label = self.scene.predict(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (self.width, self.height))

        """
        people, full = self.get_human(resize)
        for (x, y, w, h) in people:
            person = resize[y:y + h, x:x + w]
            person_resize = cv2.resize(person, (self.Resources.motion_weight, self.Resources.motion_height))
            person_image = np.array(person_resize)
            #motion = self.predict_motion(person_image, full)

            cv2.rectangle(resize, (x, y), (x + w, y + h), (0, 0, 255), 2)
        """

        if(label == 5):
            print("\t\t경기 시작 전입니다.")
        elif(label == 0):
            print("\t\t경기장을 보여주고 있습니다.")
        elif(label == 6):
            print("\t\t1루 쪽을 보여주네요.")
        elif (label == 9):
            print("\t\t3루 쪽을 보여주네요.")
        elif (label == 2):
            print("\t\t관중들이 응원을 하고 있습니다.")
        elif (label == 8):
            print("\t\t선수들이 클로즈업 되었네요.")
        elif (label == 3):
            print("\t\t"+str(relayText["batorder"])+"번 타자의 모습입니다.")
        elif (label == 4):
            print("\t\t투수, 타자 그리고 포수가 영상에 잡히네요.")
        elif (label == 1):
            print("\t\t투수의 모습입니다.")
        elif (label == 7):
            print("\t\t코치들의 모습이네요.")
        else:
            print('\t\t기타 장면 입니다.')

        print("\t\t============================")

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

    def make_motion_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.Resources.motion_weight, self.Resources.motion_height, 1])
        self.Y = tf.placeholder(tf.float32, [None, 4])
        self.keep_prob = tf.placeholder(tf.float32)

        C1_1 = conv_layer(filter_size=3, fin=1, fout=3, din=self.X, name='C1_1')
        C1_2 = conv_layer(filter_size=3, fin=3, fout=9, din=C1_1, name='C1_2')
        P1 = pool(C1_2, option="maxpool")
        P1 = tf.nn.dropout(P1, keep_prob=self.keep_prob)

        C2_1 = conv_layer(filter_size=3, fin=9, fout=27, din=P1, name='C2_1')
        C2_2 = conv_layer(filter_size=3, fin=27, fout=54, din=C2_1, name='C2_2')
        P2 = pool(C2_2, option="maxpool")
        P2 = tf.nn.dropout(P2, keep_prob=self.keep_prob)

        C3_1 = conv_layer(filter_size=3, fin=54, fout=54, din=P2, name='C3_1')
        C3_2 = conv_layer(filter_size=3, fin=54, fout=54, din=C3_1, name='C3_2')
        P3 = pool(C3_2, option="maxpool")
        P3 = tf.nn.dropout(P3, keep_prob=self.keep_prob)

        C4_1 = conv_layer(filter_size=3, fin=54, fout=54, din=P3, name='C4_1')
        C4_2 = conv_layer(filter_size=3, fin=54, fout=54, din=C4_1, name='C4_2')
        P4 = pool(C3_2, option="maxpool")
        P4 = tf.nn.dropout(P4, keep_prob=self.keep_prob)

        print(P4)

        fc0 = tf.reshape(P4, [-1, 8 * 10 * 54])

        with tf.device("/cpu:0"):
            W1 = tf.get_variable("W1", shape=[8 * 10 * 54, 4096], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([4096]))
            fc1 = tf.nn.relu(tf.matmul(fc0, W1) + b1)

            W2 = tf.get_variable("W2", shape=[4096, 4096], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([4096]))
            fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)

            W3 = tf.get_variable("W3", shape=[4096, 4], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([4]))
            self.model = tf.matmul(fc2, W3) + b3

        chk_full = './_model/action_full/action.ckpt'
        chk_upper= './_model/action_upper/action.ckpt'

        self.full = tf.Session()
        self.upper = tf.Session()

        self.saver1 = tf.train.Saver()
        self.saver2 = tf.train.Saver()

        self.saver1.restore(self.full, chk_full)
        self.saver2.restore(self.upper, chk_upper)


    def predict_motion(self, frame, full=1):
        frame = frame.reshape(-1, self.Resources.motion_weight, self.Resources.motion_height, 1)
        if(full == 1):
            result = self.full.run(tf.argmax(self.model, 1), feed_dict={self.X: frame, self.keep_prob: 1})
        else:
            result = self.upper.run(tf.argmax(self.model, 1), feed_dict={self.X: frame, self.keep_prob: 1})

        if (result == 0):
            print("\t\t누가 걷고 있습니다.")

        elif (result == 1):
            print("\t\t누가 조깅하듯이 뛰고 있네요")

        elif (result == 2):
            print("\t\t누가 달리고 있습니다.")

        elif (result == 3):
            print("\t\t누가 던지고 있습니다.")

        return result

