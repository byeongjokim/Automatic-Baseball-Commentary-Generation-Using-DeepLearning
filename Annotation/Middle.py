import time
import random
import cv2
import tensorflow as tf
import numpy as np
import owlready2
from Annotation.Rule_Annotation import RuleData
from Annotation.Scene_Annotation import SceneData
from Annotation.Ontology_String import *
from skimage.measure import compare_ssim as ssim
from NN.tinyYOLOv2.test import ObjectDetect
from NN.motion_model import Classifier

class Middle():
    frame_no = 0
    frame = []

    def __init__(self, gameName, Resources):
        print("init annotation")
        self.sess = tf.Session()
        self.resources = Resources

        owlready2.onto_path.append("_data/_owl/")
        self.onto = owlready2.get_ontology("180515SKOB.owl")
        self.onto.load()

        self.objectdetect = ObjectDetect(self.sess)

        self.ruleData = RuleData(gameName, Resources, self.onto)
        self.sceneData = SceneData(Resources, self.onto, self.sess)

        self.motion = Classifier(self.sess)

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
        h, w, c = self.resources.frame.shape
        ratio_h = h / 416
        ratio_w = w / 416
        person_bbox = {"A": [], "B": [], "C": [], "D": [], "E": [], "A_": None, "B_": None, "C_": None,
                       "D_": None, "E_": None}
        person_image = {"A": [], "B": [], "C": [], "D": [], "E": []}

        while( not self.resources.exit):
            if(ssim(self.resources.frame, frame, multichannel=True) < 0.6): #scene changed
                isPitcher = 0
                frame = self.resources.frame
                label, annotation = self.sceneData.get_Annotation(self.resources.frame)
                person_bbox = {"A": [], "B": [], "C": [], "D": [], "E": [], "A_": None, "B_": None, "C_": None, "D_": None, "E_": None}
                person_image = {"A": [], "B": [], "C": [], "D": [], "E": []}

            else:
                if(isPitcher == 3):
                    print("\t\t" + annotation)

                if(isPitcher > 10 and label == 2):
                    isPitcher = 0
                    print("\t\t" + self.sceneData.pitcher())

                if(isPitcher > 17 and label != 0):
                    isPitcher = 0
                    print("\t\t" + self.sceneData.gameinfo())

                bboxes = self.objectdetect.predict(self.resources.frame)
                if (bboxes):
                    for bbox in bboxes:
                        if (bbox[2] == 'person' and bbox[0][0] > 0 and bbox[0][1] > 0 and bbox[0][2] > 0 and bbox[0][3] > 0):
                            person_bbox, person_image = self.motion.insert_person(frame, person_bbox, person_image,
                                                                                  [bbox[0][0] * ratio_w,
                                                                                   bbox[0][1] * ratio_h,
                                                                                   bbox[0][2] * ratio_w,
                                                                                   bbox[0][3] * ratio_h]
                                                                                  )

                print(person_bbox["A_"], person_bbox["B_"], person_bbox["C_"], person_bbox["D_"], person_bbox["E_"])

                isPitcher = isPitcher + 1
        return 1
