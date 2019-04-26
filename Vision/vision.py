import cv2
import tensorflow as tf
import numpy as np
from Vision.Model.scene import Scene_Model
from Vision.Model.detect import Detect_Model
from Vision.Model.motion import Motion_Model, CAE
from Vision.Model.situation import Situation_Model
from Vision.annotation import Annotation
import time
import random

class Vision(object):
    annotation_history = []
    def __init__(self, resource=None, istest=1):
        if(istest == 1):
            sess = tf.Session()
            self.scene = Scene_Model(sess=sess, istest=istest)
            self.detect = Detect_Model(sess=sess, istest=istest)
            self.motion = Motion_Model(sess=sess, istest=istest)
            self.situation = Situation_Model(sess=sess, istest=istest)
            self.annotation = Annotation()

            self.resource = resource

    def play(self):
        time.sleep(7)

        image = self.resource.get_frame()
        self.image_width, self.image_height, self.image_channel = image.shape
        position_images_seq = {"pitcher": [], "batter": [], "player": []}
        position_images_bbox_seq = {"pitcher": [], "batter": [], "player": []}
        motion_label = {"pitcher": None, "batter": None, "player": None}

        scene_queue = []
        scene_length = 0
        human_queue = []

        pre_scene_label = -1
        count = 0
        too_long = 0
        situation_count = 0
        while(1):
            image = self.resource.get_frame()
            scene_label, scene_score, featuremap = self.scene.predict(image=image)

            if(scene_label != pre_scene_label):
                position_images_seq = {"pitcher": [], "batter": [], "player": []}
                position_images_bbox_seq = {"pitcher": [], "batter": [], "player": []}
                motion_label = {"pitcher":None, "batter":None, "player":None}
                count = 0

            human_coordinates = self.detect.predict(image=image)
            if(human_coordinates):
                position_images_seq, position_images_bbox_seq = self._get_player_seq(image=image, position_images_seq=position_images_seq, position_images_bbox_seq=position_images_bbox_seq, human_coordinates=human_coordinates)

            if (scene_label == 0 and position_images_seq["pitcher"] and motion_label["pitcher"] != 3):
                pitcher_motion_label, motion_score = self.motion.predict(position_images_seq["pitcher"])
                motion_label["pitcher"] = pitcher_motion_label
                anno = self.annotation.get_motion_annotation(scene_label=scene_label, motion_label=pitcher_motion_label, who="pitcher", resource=self.resource)
                self._choose_random_annotation(anno)
                if(pitcher_motion_label == 3):
                    situation_count = 1

            if (scene_label == 0 and position_images_seq["batter"] and motion_label["batter"] != 0):
                batter_motion_label, motion_score = self.motion.predict(position_images_seq["batter"])
                motion_label["batter"] = batter_motion_label
                #anno = self.annotation.get_motion_annotation(scene_label=scene_label, motion_label=batter_motion_label, who="batter", resource=self.resource)
                #self._choose_random_annotation(anno)

            if (position_images_seq["player"] and motion_label["player"] == None):
                player_motion_label, motion_score = self.motion.predict(position_images_seq["player"])
                motion_label["player"] = player_motion_label
                anno = self.annotation.get_motion_annotation(scene_label=scene_label, motion_label=player_motion_label, resource=self.resource)
                self._choose_random_annotation(anno)

            if(situation_count > 0):
                situation_count = situation_count + 1
                scene_queue.append(featuremap)
                scene_length = scene_length + 1
                if(human_coordinates):
                    human = []
                    for x, y, w, h, _ in human_coordinates[:6]:
                        human.append(cv2.resize(image[y:h, x:w], (50, 50)))
                    human_queue.append(human)
                else:
                    human_queue.append([])
                scene_queue, human_queue, situation_label, situation_score = self.situation.predict(scene_queue=scene_queue, human_queue=human_queue, L=scene_length)
                """
                if (situation_count > 7):
                    print("===============", situation_count, situation_label, situation_score, scene_label)
                """
                if (situation_count == 9):
                    anno = self.annotation.get_situation_annotation(situation_label, scene_label)
                    situation_count = 0
                    self._choose_random_annotation(anno)

                if(situation_count > 15):
                    situation_count = 0
                    scene_queue = []
                    scene_length = 0
                    human_queue = []

            anno = []
            if(count == 10 and scene_label != 9):
                self.annotation.reload()
                if(scene_label == 0):
                    anno = anno + self.annotation.search_batter(self.resource.get_gamecode(), self.resource.get_batter(), self.resource.get_strike_ball_out())
                    anno = anno + self.annotation.search_pitcher(self.resource.get_gamecode(), self.resource.get_pitcher(), self.resource.get_strike_ball_out())
                    anno = anno + self.annotation.search_pitcherbatter(self.resource.get_gamecode(), self.resource.get_batter(), self.resource.get_pitcher(), self.resource.get_strike_ball_out())
                    anno = anno + self.annotation.search_runner(self.resource.get_batterbox())

                elif(scene_label == 1):
                    anno = anno + self.annotation.search_batter(self.resource.get_gamecode(), self.resource.get_batter(), self.resource.get_strike_ball_out())

            if(count > 15 and scene_label == 2):
                self.annotation.reload()
                anno = anno + self.annotation.search_pitcher(self.resource.get_gamecode(), self.resource.get_pitcher(), self.resource.get_strike_ball_out())
                count = 0

            if(too_long > 20 and scene_label != 9):
                self.annotation.reload()
                anno = anno + self.annotation.search_gameInfo(self.resource.get_gamecode(), self.resource.get_inn(), self.resource.get_gamescore(), self.resource.get_gameinfo())
                anno = anno + self.annotation.search_team(self.resource.get_gameinfo(), self.resource.get_btop())
                anno = anno + self.annotation.search_batter(self.resource.get_gamecode(), self.resource.get_batter(), self.resource.get_strike_ball_out())
                anno = anno + self.annotation.search_runner(self.resource.get_batterbox())
                too_long = 0
            self._choose_random_annotation(anno)

            pre_scene_label = scene_label
            count = count + 1
            too_long = too_long + 1

    def _choose_random_annotation(self, anno):
        if not (anno == []):
            count = 0
            output = random.choice(anno)
            while(output[-7:] in self.annotation_history):
                if(count > 8):
                    break
                output = random.choice(anno)
                count = count + 1

            if(len(self.annotation_history) > 8):
                self.annotation_history.pop(0)
            self.annotation_history.append(output[-7:])
            self.resource.set_annotation(output)
            print(output)


    def _get_player_seq(self, image, position_images_seq, position_images_bbox_seq, human_coordinates):
        pitcher = [human_coordinate[:4] for human_coordinate in human_coordinates if (human_coordinate[4] == "pitcher")]
        batter = [human_coordinate[:4] for human_coordinate in human_coordinates if (human_coordinate[4] == "batter")]
        player = [human_coordinate[:4] for human_coordinate in human_coordinates if (human_coordinate[4] == "player")]

        for p in pitcher:
            if(position_images_bbox_seq["pitcher"]):
                if(self._cal_mean_iou(p, [position_images_bbox_seq["pitcher"][-1]]) > 0.3):
                    position_images_bbox_seq["pitcher"].append(p)
                    position_images_seq["pitcher"].append(image[p[1]:p[3], p[0]:p[2]])
            else:
                position_images_bbox_seq["pitcher"].append(p)
                position_images_seq["pitcher"].append(image[p[1]:p[3], p[0]:p[2]])

        for p in batter:
            if(position_images_bbox_seq["batter"]):
                if(self._cal_mean_iou(p, [position_images_bbox_seq["batter"][-1]]) > 0.3):
                    position_images_bbox_seq["batter"].append(p)
                    position_images_seq["batter"].append(image[p[1]:p[3], p[0]:p[2]])
            else:
                position_images_bbox_seq["batter"].append(p)
                position_images_seq["batter"].append(image[p[1]:p[3], p[0]:p[2]])

        if (player):
            player.sort(key=lambda x: (pow(((x[0] + x[2])/2 - self.image_width/2), 2) + pow(((x[1] + x[3])/2 - self.image_height/2), 2)))
            p = player[0]
            if(position_images_bbox_seq["player"]):
                if (self._cal_mean_iou(p, [position_images_bbox_seq["player"][-1]]) > 0.3):
                    position_images_bbox_seq["player"].append(p)
                    position_images_seq["player"].append(image[p[1]:p[3], p[0]:p[2]])
            else:
                position_images_bbox_seq["player"].append(p)
                position_images_seq["player"].append(image[p[1]:p[3], p[0]:p[2]])

        return position_images_seq, position_images_bbox_seq

    def _cal_mean_iou(self, bbox1, bboxes2):
        s = 0.0
        for bbox2 in bboxes2:
            min_x = max(bbox1[0], bbox2[0])
            max_x = min(bbox1[2], bbox2[2])
            min_y = max(bbox1[1], bbox2[1])
            max_y = min(bbox1[3], bbox2[3])

            if (max_x < min_x or max_y < min_y):
                s = s + 0.0
            else:
                inter = (max_x - min_x) * (max_y - min_y)
                bb1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                bb2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

                iou = inter / (bb1 + bb2 - inter)
                if (iou >= 0.0 and iou <= 1.0):
                    s = s + iou
                else:
                    s = s + 0.0
        return s / len(bboxes2)

    def train(self, model=None):
        sess = tf.Session()
        if(model == "scene"):
            scene = Scene_Model(sess=sess, istest=0)
            scene.load_data(["20180929_531801"])
            scene.train()
        elif(model == "detect"):
            return 1
        elif(model == "motion"):
            motionCAE = CAE()
            motionCAE.train()
            # motion = Motion_Model(sess=sess, istest=0)