# -*- coding: utf-8 -*-
import cv2
from NN.scene_model import Scene_Model
from Annotation.SceneString import *
from Annotation.Ontology_data import *

class SceneData():

    prev = -1

    def __init__(self, Resources, onto, shape=(320, 180)):
        print("init_sceneData")
        self.Resources = Resources
        self.onto = onto

        self.width = shape[0]
        self.height = shape[1]

        self.scene = Scene_Model()
        self.scene.make_model()

    def get_Annotation(self, frame, t=0.7):
        #print("\t\t\t\t대기시간이 길어 영상처리로 텍스트 생성")

        label, score = self.scene.predict(frame)
        #print("점수 : " + str(score * 100) + "%")
        if(score > t):
            if(self.prev != label):
                annotation = ""
                if (label == 0):
                    annotation = self.batterBox()

                elif(label == 1):
                    annotation = self.player("batter")

                elif (label == 2):
                    annotation = self.player("1")

                elif (label == 3):
                    annotation = self.coach()

                elif (label == 4):
                    annotation = self.gallery()

                elif (label == 5):
                    annotation = self.firstBase()

                elif (label == 6):
                    annotation = self.centerOutField()

                elif (label == 7):
                    annotation = self.rightOutField()

                elif (label == 8):
                    annotation = self.secondBase()

                elif (label == 9):
                    annotation = self.etc()

                elif (label == 10):
                    annotation = self.thirdBase()

                elif (label == 11):
                    annotation = self.leftOutField()

                elif (label == 12):
                    annotation = self.sS()

                self.prev = label
                print("\t\t\t"+annotation)

        return label


    def batterBox(self):
        return BatterBox()

    def coach(self):
        return coach()

    def gallery(self):
        return gallery()



    def leftOutField(self):
        return OutField("left")

    def centerOutField(self):
        return OutField("center")

    def rightOutField(self):
        return OutField("right")



    def firstBase(self):
        return Base("1")

    def secondBase(self):
        return Base("2")

    def thirdBase(self):
        return Base("3")

    def sS(self):
        return Player("ss")

    def player(self, player):
        return Player(player)



    def etc(self):
        return etc()








