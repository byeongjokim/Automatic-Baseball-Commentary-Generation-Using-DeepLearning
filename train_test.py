import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from NN.Make_data import Make_SceneData

from NN.scene_model import Scene_Model
from NN.motion_model import Motion_model

def train_scene():
    s = Scene_Model()
    s.load_data()

    s.make_model()
    s.train()
    s.test()

def test_scene():
    s = Scene_Model()
    s.num_label = 10
    image = cv2.imread("./scene_data/test/20171029KIADUSAN/closeup_25.jpg")
    s.make_model()
    s.predict(image)

def train_motion():
    m = Motion_model()
    m.load_data()
    m.make_model()
    m.train()
    m.test()

def test_motion():
    return 1

def make_scene_data():
    s = Make_SceneData("./_data/20171029KIADUSAN.mp4")
    s.save_image_with_frame_interval()



#train_scene()
#test_scene()
train_motion()

#make_scene_data()