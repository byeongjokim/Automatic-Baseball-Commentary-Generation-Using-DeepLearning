import cv2
import os
import argparse
#from NN.Make_data import Make_SceneData
from NN.scene_model import Scene_Model
from NN.motion_model import CAE, Classifier
import tensorflow as tf
from NN.tinyYOLOv2.test import ObjectDetect

def train_scene(play):
    sess =tf.Session()
    s = Scene_Model(sess)
    play = ["180401HTLG", "180401NCLT", "180401OBKT", "180401SKHH", "180401WOSS",
            "180403HTSK", "180403KTWO", "180403LGOB", "180403LTHH", "180403SSNC",
            "180404HTSK", "180404KTWO", "180404LGOB", "180404LTHH", "180404SSNC",
            "180405KTWO", "180405SSNC",
            "180406LGLT", "180406WOHT",
            "180407HHKT", "180407LGLT", "180407NCOB", "180407SSSK", "180407WOHT",
            "180408HHKT", "180408LGLT", "180408NCOB", "180408SSSK", "180408WOHT",
            "180410HTHH", "180410KTNC", "180410OBSS", "180410SKLG", "180410WOLT",
            "180411HTHH", "180411KTNC", "180411OBSS", "180411SKLG", "180411WOLT",
            "180412HTHH", "180412KTNC", "180412OBSS", "180412SKLG", "180412WOLT",
            "180413KTLG", "180413LTHT", "180413NCSK", "180413OBWO", "180413SSHH"
            ]

    s.load_data(play)

    s.make_model()
    s.train()
    s.test()

def test_scene(image_name, t):
    sess = tf.Session()
    s = Scene_Model(sess)

    image = cv2.imread(image_name)
    s.make_model()
    print(s.predict(image))

#train_scene(1)
#test_scene("_data/180401NCLT/186.jpg", 0)

#make_scene_data()

def get_motion_data(v, count, start=1000, interval=5):
    s_count = count

    sess = tf.Session()

    o = ObjectDetect(sess)

    video = cv2.VideoCapture("_data/"+v+"/"+v+".mp4")

    video.set(1, start)
    success, img = video.read()
    if not success:
        return 1

    h, w, c = img.shape
    ratio_h = h / 416
    ratio_w = w / 416


    while True:
        video.set(1, start)
        success, img = video.read()
        if not success:
            break
        if(count - s_count > 500):
            break

        bboxes = o.predict(img)
        if bboxes:
            for bbox in bboxes:
                if(bbox[2] == 'person'):
                    b = img[int(bbox[0][1] * ratio_h): int(bbox[0][3] * ratio_h),
                        int(bbox[0][0] * ratio_w): int(bbox[0][2] * ratio_w)]
                    cv2.imwrite("_data/_motion/" + str(count) + ".jpg", b)
                    count = count + 1

        start = start + interval

    return count

videos = ["180401HTLG", "180401NCLT", "180401OBKT", "180401SKHH", "180401WOSS",
            "180403HTSK", "180403KTWO", "180403LGOB", "180403LTHH", "180403SSNC",
            "180404HTSK", "180404KTWO", "180404LGOB", "180404LTHH", "180404SSNC",
            "180405KTWO", "180405SSNC",
            "180406LGLT", "180406WOHT"]
"""
            "180407HHKT", "180407LGLT", "180407NCOB", "180407SSSK", "180407WOHT",
            "180408HHKT", "180408LGLT", "180408NCOB", "180408SSSK", "180408WOHT",
            "180410HTHH", "180410KTNC", "180410OBSS", "180410SKLG", "180410WOLT",
            "180411HTHH", "180411KTNC", "180411OBSS", "180411SKLG", "180411WOLT",
            "180412HTHH", "180412KTNC", "180412OBSS", "180412SKLG", "180412WOLT",
            "180413KTLG", "180413LTHT", "180413NCSK", "180413OBWO", "180413SSHH"
"""

#count = 0
#for v in videos:
#    print(count)
#    count = get_motion_data(v, count, start=1000, interval=5)


sess = tf.Session()
#m= CAE(sess)
#m.train()
#m.test()

m = Classifier(sess)
m.model()
#m.train()
m.test()

"""
sess = tf.Session()
m = Motion(sess)
m.model()
#m.train()
img = cv2.imread("_data/motion/_motion/32906.jpg")
m.test(img)


sess = tf.Session()

o = ObjectDetect(sess)
scene = Scene_Model(sess)
scene.make_model()

m = Motion(sess)
m.model()
#m.train()

img = cv2.imread("_data/180401HTLG/2342.jpg")
#img = cv2.imread("_data/180401HTLG/1829.jpg")
#img = cv2.imread("_data/180401HTLG/1767.jpg")
#img = cv2.imread("_data/180401HTLG/1090.jpg")
#img = cv2.imread("_data/180401HTLG/473.jpg")
#img = cv2.imread("_data/180401HTLG/487.jpg")
#img = cv2.imread("_data/180401HTLG/377.jpg")


label, score = scene.predict(img)
print(label)

h, w, c = img.shape
ratio_h = h / 416
ratio_w = w / 416

bboxes = o.predict(img)
if(bboxes):
    for bbox in bboxes:
        print(bbox)
        if(bbox[2] == 'person'  and bbox[0][0] > 0 and bbox[0][1] > 0 and bbox[0][2] > 0 and bbox[0][3] > 0 ):
            b = img[int(bbox[0][1] * ratio_h): int(bbox[0][3] * ratio_h),
                int(bbox[0][0] * ratio_w): int(bbox[0][2] * ratio_w)]
            motion = m.test(b)

            img = cv2.rectangle(img, (int(bbox[0][0] * ratio_w), int(bbox[0][1] * ratio_h)), (int(bbox[0][2] * ratio_w), int(bbox[0][3] * ratio_h)), (255, 0, 0), 2)
            img = cv2.putText(img, motion, (int(bbox[0][0] * ratio_w), int(bbox[0][1] * ratio_h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

#img = cv2.putText(img, label, (0, h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
cv2.imshow("result", img)
cv2.waitKey(0)
"""