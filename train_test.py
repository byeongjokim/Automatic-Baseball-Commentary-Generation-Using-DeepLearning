import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from NN.Make_data import Make_SceneData

from NN.scene_model import Scene_Model
from NN.motion_model import Motion_model
from NN.person_model import Person_model

def train_scene():
    s = Scene_Model()
    s.load_data()

    s.make_model()
    s.train()
    s.test()

def test_scene():
    s = Scene_Model()

    image = cv2.imread("_data/180407LGLT/510.jpg")
    s.make_model()
    print(s.predict(image))

def make_scene_data():
    l = ["doru1"]
    s = Make_SceneData()

    for p in l:
        s.set_path(p)
        s.save_image_with_frame_interval(start=0)

def train_motion():
    m = Motion_model()
    m.load_data()
    m.CNN_pretrain()
    m.CNN_train()
    m.make_model()
    m.train()
    m.test()

def test_motion():
    return 1

def train_person():
    p = Person_model()
    p.load_data()



#train_scene()
test_scene()


#train_motion()
#train_person()

#make_scene_data()