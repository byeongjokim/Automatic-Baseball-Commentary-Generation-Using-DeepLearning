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
import queue

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

        num_prev_annotation = 10
        self.prev_annotaion = queue.Queue(num_prev_annotation)

    def generate_Annotation_with_Rule(self, count_delta, fps, o_start):
        self.ruleData.set_Start(count_delta, fps, o_start)
        self.ruleData.get_Annotation()
        return 1

    def generate_Annotation_with_Scene(self):
        counter = 0

        time.sleep(5)

        pre_label = -1
        h, w, c = self.resources.frame.shape
        frame = np.zeros((h, w, c), dtype=np.uint8)
        position = Position(motion=self.motion, frame_shape=(h, w), resource=self.resources)

        while( not self.resources.exit ):
            label, score, annotation = self.sceneData.get_Annotation(self.resources.frame)
            #print(label, counter)

            """
            if(score > 0.8):
                self.resources.set_annotation_2(label)
            """

            if(label != pre_label and ssim(self.resources.frame, frame, multichannel=True) < 0.6): #scene changed
                #print("refresh")
                counter = 0
                frame = self.resources.frame
                position.clear()

            if(counter == 7 and annotation):
                annotation = self.get_random_annotation(annotation)

                if("안익훈65115" in annotation):
                    annotation.replace("안익훈65115", "이형종78135")
                print("from scene \t\t" + annotation)
                self.resources.set_annotation(annotation)

            if(counter > 24):
                if(label == 2):
                    pitcher_annotation = self.sceneData.pitcher()
                    pitcher_annotation = self.get_random_annotation(pitcher_annotation)

                    print("from pitcher \t\t" + pitcher_annotation)
                    self.resources.set_annotation(pitcher_annotation)
                else:
                    gameinfo_annotation = self.sceneData.gameinfo()
                    gameinfo_annotation = self.get_random_annotation(gameinfo_annotation)

                    print("from gameinfo \t\t" + gameinfo_annotation)
                    self.resources.set_annotation(gameinfo_annotation)

                counter = 0

            bboxes = self.detect.predict(self.resources.frame)
            if (bboxes):
                position.insert_person(self.resources.frame, bboxes, label)

            motion_annotation = position.annotation(label, position.get_bbox())
            if(motion_annotation):
                motion_annotation = self.get_random_annotation(motion_annotation)

                print("from motion\t\t"+ motion_annotation)
                self.resources.set_annotation(motion_annotation)
            pre_label = label
            counter = counter + 1

        return 1

    def get_random_annotation(self, annotation):
        #print(list(self.prev_annotaion.queue))
        counter = 0
        while(1):
            print(counter)
            output = random.choice(annotation)
            if counter > 5:
                if (self.prev_annotaion.full()):
                    self.prev_annotaion.get_nowait()
                self.prev_annotaion.put(output)
                return output
            if not (output in list(self.prev_annotaion.queue)):
                if(self.prev_annotaion.full()):
                    self.prev_annotaion.get_nowait()
                self.prev_annotaion.put(output)
                return output

            counter = counter + 1

