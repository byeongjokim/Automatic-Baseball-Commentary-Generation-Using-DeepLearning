# -*- coding: utf-8 -*-
import cv2
import random
from NN.scene_model import Scene_Model
from Annotation.Ontology_String import *


class SceneData():

    prev = -1
    num_prev_annotation = 5
    prev_annotaion = []

    def __init__(self, Resources, onto, sess, shape=(320, 180)):
        print("init_sceneData")
        self.resources = Resources
        self.onto = onto

        self.width = shape[0]
        self.height = shape[1]

        self.scene = Scene_Model(sess)
        self.scene.make_model()

        self.timer = 0
        for i in range(self.num_prev_annotation):
            self.prev_annotaion.append("0")

    def get_Annotation(self, frame, t=0.7):
        self.timer = self.timer + 1
        #print("\t\t\t\t대기시간이 길어 영상처리로 텍스트 생성")
        annotation = ''
        label, score = self.scene.predict(frame)
        #print("점수 : " + str(score * 100) + "%")
        if(score > t):
            if(self.prev != label):
                annotation = ""
                if (label == 0): #pitchingbatting
                    annotation = self.batterBox()

                elif (label == 1):
                    annotation = self.batter()

                elif (label == 3): #coach
                    annotation = self.coach()

                elif (label == 4): #gallery
                    annotation = self.gallery()

                elif (label == 5):
                    annotation = self.first()

                else:
                    annotation = ""

                self.prev = label
                #print("\t\t"+annotation)

        return label, annotation


    def batterBox(self):

        gameCode = self.resources.get_gamecode()
        b = self.resources.get_batter()
        p = self.resources.get_pitcher()
        annotation = []
        annotation = annotation + search_batter(gameCode, b)
        annotation = annotation + search_pitcher(gameCode, p)
        annotation = annotation + search_pitcherbatter(gameCode, p, b)
        annotation = annotation + search_gameInfo(gameCode, self.resources.get_inn(), self.resources.get_gamescore(), self.resources.get_gameinfo())

        return self.get_random_annotation(annotation)


    def pitcher(self):
        gameCode = self.resources.get_gamecode()
        p = self.resources.get_pitcher()
        annotation = search_pitcher(gameCode, p)

        return self.get_random_annotation(annotation)

    def batter(self):
        gameCode = self.resources.get_gamecode()
        b = self.resources.get_batter()
        annotation = search_batter(gameCode, b)

        return self.get_random_annotation(annotation)

    def gallery(self):
        return "관개ㅐ애애애애애애애애애애액"

    def coach(self):
        return "코치이이이이이이이이이ㅣㅇ"
    
    def first(self):
        return "1루우우우우우우우우ㅜ우우웅"

    def gameinfo(self):
        annotation = search_gameInfo(self.resources.get_gamecode(), self.resources.get_inn(), self.resources.get_gamescore(),
                        self.resources.get_gameinfo())

        return self.get_random_annotation(annotation)

    def get_random_annotation(self, annotation):
        output = random.choice(annotation)

        count = 0
        while(output in self.prev_annotaion):
            if(count > self.num_prev_annotation):
                break
            output = random.choice(annotation)
            count = count + 1

        self.prev_annotaion.pop()
        self.prev_annotaion.insert(0, output)

        return output