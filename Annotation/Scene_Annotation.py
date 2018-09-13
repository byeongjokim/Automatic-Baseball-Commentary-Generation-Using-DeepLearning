# -*- coding: utf-8 -*-
import cv2
import random
from NN.scene_model import Scene_Model
from Annotation.Ontology_String import *


class SceneData():

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

        annotation = ''
        label, score = self.scene.predict(frame)

        if(score > t):

            if (label == 0): #pitchingbatting
                annotation = self.batterBox()

            elif (label == 1):
                annotation = self.batter()

            else:
                annotation = None

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