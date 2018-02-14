import cv2
import os
import csv
import tensorflow as tf
import numpy as np
from imutils.object_detection import non_max_suppression

class Motion_model():

    width = 224
    height = 224

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

        train_data = []
        for i in dataset:
            image_set = []
            for j in range(int(i["start"]), int(i["end"])+1):
                image_set.append(
                                    cv2.resize(
                                        cv2.cvtColor(
                                            cv2.imread(folder_path + str(j) + ".jpg"),
                                            cv2.COLOR_BGR2GRAY
                                        ),
                                        (self.width, self.height)
                                    )
                                 )
            sett = {"image" : image_set, "label" : i["label"]}
            train_data.append(sett)

        print(train_data[0]["image"])
        print(train_data[0]["label"])

        _y = np.array([i["label"] for i in train_data])
        y = np.zeros((len(_y), len(set(_y))))
        y[np.arange(len(_y)), [self.kind_motion.index(i) for i in _y]] = 1

        self.num_label = len(set(_y))

        self.Y = y
        return 1

    def make_model(self):
        X = tf.placeholder(tf.float32, [None, self.width, self.height, 1])
        Y = tf.placeholder(tf.float32, [None, self.num_label])
        self.keep_prob = tf.placeholder(tf.float32)

        


        return 1

    def train(self):
        return 1

    def test(self):
        return 1

    def predict(self, image):
        return 1