import time
import random
import tensorflow as tf
import numpy as np
import owlready2
from Annotation.Rule_Annotation import RuleData
from Annotation.Scene_Annotation import SceneData
from Annotation.Ontology_String import *
from skimage.measure import compare_ssim as ssim
from NN.tinyYOLOv2.test import ObjectDetect

class Middle():
    frame_no = 0
    frame = []

    def __init__(self, gameName, Resources):
        print("init annotation")
        self.sess = tf.Session()
        self.resources = Resources

        self.objectdetect = ObjectDetect(self.sess)

        owlready2.onto_path.append("_data/_owl/")
        self.onto = owlready2.get_ontology("180515SKOB.owl")
        self.onto.load()

        self.ruleData = RuleData(gameName, Resources, self.onto)
        self.sceneData = SceneData(Resources, self.onto, self.sess)




    def generate_Annotation_with_Rule(self, count_delta, fps, o_start):
        self.ruleData.set_Start(count_delta, fps, o_start)
        self.ruleData.get_Annotation()
        return 1

    def generate_Annotation_with_Scene(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        isPitcher = 0

        time.sleep(3)
        print("Scene model 시작")
        label = -1
        annotation = ""
        while( not self.resources.exit):
            if(ssim(self.resources.frame, frame, multichannel=True) < 0.6):
                isPitcher = 0
                frame = self.resources.frame
                label, annotation = self.sceneData.get_Annotation(self.resources.frame)
                #self.objectdetect.predict(frame)

            else:
                print(isPitcher, label)
                if(isPitcher == 3):
                    print("\t\t" + annotation)

                if(isPitcher > 10 and label == 2):
                    isPitcher = 0
                    print("\t\t" + self.sceneData.pitcher())

                if(isPitcher > 17 and label != 0):
                    isPitcher = 0
                    print("\t\t" + self.sceneData.gameinfo())

                isPitcher = isPitcher + 1
        return 1
