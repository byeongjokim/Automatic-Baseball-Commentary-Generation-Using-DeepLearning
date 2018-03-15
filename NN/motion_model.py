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

    cnn_batch_size = 20
    cnn_chk = './_model/motion/cnn/cnn.ckpt'
    cnn_ckpt = tf.train.get_checkpoint_state(("./_model/motion/cnn/"))

    kind_motion = ["pitching", "swinging"]
    num_label = len(kind_motion)

    def __init__(self):
        print("init motion model")

    def load_data(self):
        return 1

    def make_model(self):
        return 1

    def train(self):
        return 1

    def test(self):
        return 1

    def predict(self, image):
        return 1