# -*- coding: utf-8 -*-
import cv2
import random
from NN.scene_model import Scene_Model
from Annotation.Ontology_String import Ontology_String


class SceneData():

    num_prev_annotation = 5
    prev_annotaion = []

    def __init__(self, Resources, onto, sess, shape=(320, 180)):
        print("init_sceneData")
        self.resources = Resources
        self.onto = onto
        self.Ontology_String = Ontology_String()

        self.width = shape[0]
        self.height = shape[1]

        self.scene = Scene_Model(sess, istest=1)

        self.timer = 0
        for i in range(self.num_prev_annotation):
            self.prev_annotaion.append("0")

    def get_Annotation(self, frame, t=0.7):
        self.timer = self.timer + 1

        annotation = ''
        label, score = self.scene.predict(frame)
        #print(label, score)

        if(score > t):

            if (label == 0): #pitchingbatting
                annotation = self.batterBox()

            elif (label == 1):
                annotation = self.batter()

            elif (label == 4):
                print("관개애애애애애앵애애애애액")

            else:
                annotation = None

        return label, annotation

    def batterBox(self):

        gameCode = self.resources.get_gamecode()
        b = self.resources.get_batter()
        p = self.resources.get_pitcher()
        annotation = []
        annotation = annotation + self.Ontology_String.search_batter(gameCode, b)
        annotation = annotation + self.Ontology_String.search_pitcher(gameCode, p)
        annotation = annotation + self.Ontology_String.search_pitcherbatter(gameCode, p, b)
        annotation = annotation + self.Ontology_String.search_runner(self.resources.get_batterbox())
        annotation = annotation + self.Ontology_String.search_gameInfo(gameCode, self.resources.get_inn(), self.resources.get_gamescore(), self.resources.get_gameinfo())

        return self.get_random_annotation(annotation)

    def pitcher(self):
        gameCode = self.resources.get_gamecode()
        p = self.resources.get_pitcher()
        annotation = self.Ontology_String.search_pitcher(gameCode, p)

        return self.get_random_annotation(annotation)

    def batter(self):
        gameCode = self.resources.get_gamecode()
        b = self.resources.get_batter()
        annotation = self.Ontology_String.search_batter(gameCode, b)

        return self.get_random_annotation(annotation)

    def gameinfo(self):
        annotation = self.Ontology_String.search_gameInfo(self.resources.get_gamecode(), self.resources.get_inn(), self.resources.get_gamescore(),
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