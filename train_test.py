import cv2
import os
import argparse
#from NN.Make_data import Make_SceneData
from NN.scene_model import Scene_Model
from NN.motion_model import CAE, Classifier
import tensorflow as tf
from NN.tinyYOLOv2.test import ObjectDetect
import numpy as np


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

def get_motion_data(o, s, v, count, start=1000, interval=5):
    s_count = count



    video = cv2.VideoCapture("_data/"+v+"/"+v+".mp4")

    video.set(1, start)
    success, img = video.read()
    if not success:
        return 1

    h, w, c = img.shape
    ratio_h = h / 416
    ratio_w = w / 416

    person_bbox = {"A": [], "B": [], "C": [], "D": [], "E": [], "A_": None, "B_": None, "C_": None, "D_": None, "E_": None}
    person_image = {"A": [], "B": [], "C": [], "D": [], "E": []}

    pre_label = -1
    while True:
        video.set(1, start)
        success, frame = video.read()
        if not success:
            break
        if(count - s_count > 500):
            break

        label, score = s.predict(frame)
        if (label != pre_label):  # scene changed
            index = ["A", "B", "C", "D", "E"]

            for i in index:
                if person_bbox[i]:
                    for p in person_image[i]:
                        cv2.imwrite("_data/_motion/" + str(count) + ".jpg", p)
                        count = count + 1
            person_bbox = {"A": [], "B": [], "C": [], "D": [], "E": [], "A_": None, "B_": None, "C_": None, "D_": None, "E_": None}
            person_image = {"A": [], "B": [], "C": [], "D": [], "E": []}

        bboxes = o.predict(frame)
        if (bboxes):
            for bbox in bboxes:
                if (bbox[2] == 'person' and bbox[0][0] > 0 and bbox[0][1] > 0 and bbox[0][2] > 0 and bbox[0][3] > 0):
                    # person.append([int(((bbox[0][0] + bbox[0][2])*ratio_w)/2), int(((bbox[0][1] + bbox[0][3])*ratio_h)/2)])

                    person_bbox, person_image = insert_person(1, frame, person_bbox, person_image,
                                                              [bbox[0][0] * ratio_w, bbox[0][1] * ratio_h,
                                                               bbox[0][2] * ratio_w, bbox[0][3] * ratio_h])

        start = start + interval
        #cv2.imshow("a", frame)
        pre_label = label

        #if cv2.waitKey(1) == ord('q'):
        #    break

    return count

def train_motion():
    sess = tf.Session()
    m = Classifier(sess, istest=0)
    m.train()
    #dataset = m.load_data()
    #train_x = np.array([i["X"] for i in dataset])
    #print(train_x[0].shape)

#train_motion()



def motion_classify(v):
    person_bbox = {"A": [], "B": [], "C": [], "D": [], "E": [], "A_": None, "B_": None, "C_": None, "D_": None, "E_": None}
    person_image = {"A": [], "B": [], "C": [], "D": [], "E": []}
    sess = tf.Session()
    m = Classifier(sess)
    o = ObjectDetect(sess)
    s = Scene_Model(sess)
    s.make_model()

    video = cv2.VideoCapture(v)
    #video.set(cv2.CAP_PROP_FPS, 100)

    count = 1000

    video.set(1, 0)
    success, frame = video.read()
    h, w, c = frame.shape
    ratio_h = h / 416
    ratio_w = w / 416

    pre_label = -1
    while True:
        video.set(1, count)
        success, frame = video.read()
        if not success:
            break

        label, score = s.predict(frame)

        if(label != pre_label): #scene changed
            person_bbox = {"A": [], "B": [], "C": [], "D": [], "E": [], "A_":None, "B_":None, "C_":None, "D_":None, "E_":None}
            person_image = {"A": [], "B": [], "C": [], "D": [], "E": []}

        bboxes = o.predict(frame)
        if (bboxes):
            for bbox in bboxes:
                if (bbox[2] == 'person' and bbox[0][0] > 0 and bbox[0][1] > 0 and bbox[0][2] > 0 and bbox[0][3] > 0):
                    person_bbox, person_image = insert_person(m, frame, person_bbox, person_image, [bbox[0][0] * ratio_w, bbox[0][1] * ratio_h, bbox[0][2] * ratio_w, bbox[0][3] * ratio_h])


        frame = draw_box(frame, person_bbox)
        cv2.imshow("a", frame)
        pre_label = label

        if cv2.waitKey(1) == ord('q'):
            break
        count = count + 3

def draw_box(frame, person_bbox):
    index = ["A", "B", "C", "D", "E"]

    for i in index:
        if person_bbox[i]:
            bbox = person_bbox[i][-1]
            frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            frame = cv2.putText(frame, person_bbox[i+"_"], (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

def insert_person(m, frame, person_bbox, person_image, bbox):
    index = ["A", "B", "C", "D", "E"]

    for i in index:
        p = frame[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]
        if not (person_bbox[i]):
            person_bbox[i].append(bbox)
            person_image[i].append(p)
            break

        else:
            iou = cal_mean_iou(bbox, [person_bbox[i][-1]])
            if( 0.5  < iou ):
                person_bbox[i].append(bbox)
                person_image[i].append(p)

                person_bbox[i + "_"] = m.test(person_image[i])
                break
            else:
                pass

    return person_bbox, person_image

def cal_mean_iou(bbox1, bboxes2):
    s = 0.0
    for bbox2 in bboxes2:
        min_x = max(bbox1[0], bbox2[0])
        max_x = min(bbox1[2], bbox2[2])
        min_y = max(bbox1[1], bbox2[1])
        max_y = min(bbox1[3], bbox2[3])

        if(max_x < min_x or max_y < min_y):
            s = s + 0.0
        else:
            inter = (max_x - min_x) * (max_y - min_y)
            bb1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            bb2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

            iou = inter / (bb1 + bb2 - inter)
            if(iou >= 0.0 and iou <= 1.0):
                s = s + iou
            else:
                s = s + 0.0
    return s/len(bboxes2)


#print(cal_mean_iou([744.6153846153846, 252.6923076923077, 947.6923076923077, 503.6538461538462], [[747.6923076923077, 289.03846153846155, 960.0, 515.7692307692308]]))
motion_classify("_data/180401HTLG/180401HTLG.mp4")
#motion_classify("_data/motion_test.mp4")
"""
videos = [
            "180401HTLG", "180401NCLT", "180401OBKT", "180401SKHH", "180401WOSS",
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

sess = tf.Session()

o = ObjectDetect(sess)
s = Scene_Model(sess)
s.make_model()
count = 1848
for v in videos:
    print(count)
    count = get_motion_data(o, s, v, count, start=1000, interval=1)
"""
