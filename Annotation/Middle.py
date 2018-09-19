import time
import random
import cv2
import tensorflow as tf
import numpy as np
import owlready2
from Annotation.Rule_Annotation import RuleData
from Annotation.Scene_Annotation import SceneData
from Annotation.position import Position
from Annotation.Ontology_String import *
from skimage.measure import compare_ssim as ssim
from NN.detect_model import Detect_Model
from NN.motion_model import Classifier, TRN

class Middle():
    frame_no = 0
    frame = []

    def __init__(self, gameName, Resources):
        print("init annotation")
        self.sess = tf.Session()
        self.resources = Resources

        owlready2.onto_path.append("_data/_owl/")
        self.onto = owlready2.get_ontology("baseball.owl")
        self.onto.load()

        self.ruleData = RuleData(gameName, Resources, self.onto)

        self.motion = Classifier(self.sess, istest=1)
        self.sceneData = SceneData(Resources, self.onto, self.sess)
        self.detect = Detect_Model(self.sess, istest=1)

    def generate_Annotation_with_Rule(self, count_delta, fps, o_start):
        self.ruleData.set_Start(count_delta, fps, o_start)
        self.ruleData.get_Annotation()
        return 1

    def generate_Annotation_with_Scene(self):
        counter = 0

        time.sleep(3)

        pre_label = -1
        h, w, c = self.resources.frame.shape
        frame = np.zeros((h, w, c), dtype=np.uint8)
        position = Position(motion=self.motion, frame_shape=(h, w), resource=self.resources)

        while( not self.resources.exit ):
            label, annotation = self.sceneData.get_Annotation(self.resources.frame)
            #print(counter)

            if(label != pre_label and ssim(self.resources.frame, frame, multichannel=True) < 0.6): #scene changed
                counter = 0
                frame = self.resources.frame
                position.clear()

            if(counter == 5 and annotation):
                print("\t\t" + annotation)

            if(counter > 10 and label == 2):
                counter = 0
                print("\t\t" + self.sceneData.pitcher())

            if(counter > 10):
                counter = 0
                print("\t\t" + self.sceneData.gameinfo())

            bboxes = self.detect.predict(self.resources.frame)
            if (bboxes):
                position.insert_person(self.resources.frame, bboxes, label)

            #position.print_bbox__()
            position.annotation(label, position.get_bbox())

            pre_label = label
            counter = counter + 1
        return 1
