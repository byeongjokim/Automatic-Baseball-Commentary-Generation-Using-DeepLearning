import cv2
import os
import argparse
#from NN.Make_data import Make_SceneData
from NN.scene_model import Scene_Model
from NN.motion_model import Motion
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

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--videos', nargs='+', help='Input Video name')
    parser.add_argument('-i', '--image', help='Input Image Name in Test Model')

    parser.add_argument('-m', '--make', help='Train Model', action='store_true')
    parser.add_argument('-t', '--train', help='Train Model', action='store_true')
    parser.add_argument('-T', '--test', help='test Model', action='store_true')

    parser.add_argument('--threshold', help='Threshold in Test Model')

    args = parser.parse_args()

    if(args.make):
        make_scene_data(args.video)

    if(args.train):
        train_scene(args.video)

    if(args.test):
        test_scene(args.image, args.threshold)

'''

#train_scene(1)
#test_scene("_data/180401NCLT/186.jpg", 0)

#make_scene_data()

sess = tf.Session()

o = ObjectDetect(sess)
scene = Scene_Model(sess)
scene.make_model()

m = Motion(sess)
m.model()
#m.train()

img = cv2.imread("_data/180406WOHT/625.jpg")
cv2.imshow("aa", img)
cv2.waitKey(0)
label, score = scene.predict(img)
print(label)

h, w, c = img.shape
ratio_h = h / 416
ratio_w = w / 416

bboxes = o.predict(img)

for bbox in bboxes:
    if(bbox[2] == 'person'):
        b = img[int(bbox[0][1] * ratio_h): int(bbox[0][3] * ratio_h),
            int(bbox[0][0] * ratio_w): int(bbox[0][2] * ratio_w)]
        m.test(b)
        cv2.imshow("a", b)
        cv2.waitKey(0)
