# -*- coding: utf-8 -*-
import cv2
import queue
import random
from NN.scene_model import Scene_Model
from Annotation.Ontology_String import Ontology_String


class SceneData():
    def __init__(self, Resources, onto, sess, shape=(320, 180)):
        print("init_sceneData")
        self.resources = Resources
        self.onto = onto
        self.Ontology_String = Ontology_String()

        self.width = shape[0]
        self.height = shape[1]

        self.scene = Scene_Model(sess, istest=1)

        num_prev_annotation = 5
        self.prev_annotaion = queue.Queue(num_prev_annotation)

    def get_Annotation(self, frame, t=0.8):
        label, score = self.scene.predict(frame)
        #print(label, score)

        if(score > t):

            if (label == 0): #pitchingbatting
                annotation = self.batterBox()

            elif (label == 1):
                annotation = self.batter()

            #elif (label == 4):
            #    print("관개애애애애애앵애애애애액")

            else:
                annotation = None
        else:
            annotation = None

        return label, annotation

    def batterBox(self):

        gameCode = self.resources.get_gamecode()
        b = self.resources.get_batter()
        p = self.resources.get_pitcher()
        strike_ball_out = self.resources.get_strike_ball_out()
        annotation = []
        annotation = annotation + self.Ontology_String.search_batter(gameCode, b, strike_ball_out)
        annotation = annotation + self.Ontology_String.search_pitcher(gameCode, p, strike_ball_out)
        annotation = annotation + self.Ontology_String.search_pitcherbatter(gameCode, p, b, strike_ball_out)
        annotation = annotation + self.Ontology_String.search_runner(self.resources.get_batterbox())
        #annotation = annotation + self.Ontology_String.search_gameInfo(gameCode, self.resources.get_inn(), self.resources.get_gamescore(), self.resources.get_gameinfo())

        return self.get_random_annotation(annotation)

    def pitcher(self):
        gameCode = self.resources.get_gamecode()
        p = self.resources.get_pitcher()
        strike_ball_out = self.resources.get_strike_ball_out()
        annotation = self.Ontology_String.search_pitcher(gameCode, p, strike_ball_out)

        return self.get_random_annotation(annotation)

    def batter(self):
        gameCode = self.resources.get_gamecode()
        b = self.resources.get_batter()
        strike_ball_out = self.resources.get_strike_ball_out()
        annotation = self.Ontology_String.search_batter(gameCode, b, strike_ball_out)

        return self.get_random_annotation(annotation)

    def gameinfo(self):
        annotation = self.Ontology_String.search_gameInfo(self.resources.get_gamecode(), self.resources.get_inn(), self.resources.get_gamescore(),
                        self.resources.get_gameinfo())

        return self.get_random_annotation(annotation)

    def get_random_annotation(self, annotation):
        while(1):
            output = random.choice(annotation)
            if(output in list(self.prev_annotaion.queue)):
                continue
            else:
                if(self.prev_annotaion.full()):
                    self.prev_annotaion.get()
                self.prev_annotaion.put(output)
                return output
