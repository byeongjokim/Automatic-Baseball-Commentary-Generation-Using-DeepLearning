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

def motion_classify(v):
    person = {"A": [], "B": [], "C": [], "D": [], "E": [], "A_": None, "B_": None, "C_": None, "D_": None, "E_": None}

    sess = tf.Session()
    m = Classifier(sess)
    o = ObjectDetect(sess)
    s = Scene_Model(sess)
    s.make_model()

    video = cv2.VideoCapture(v)
    #video.set(cv2.CAP_PROP_FPS, 100)

    success, frame = video.read()
    h, w, c = frame.shape
    ratio_h = h / 416
    ratio_w = w / 416

    pre_label = -1
    while True:
        success, frame = video.read()
        if not success:
            break

        label, score = s.predict(frame)

        if(label != pre_label): #scene changed
            person = {"A": [], "B": [], "C": [], "D": [], "E": [], "A_":None, "B_":None, "C_":None, "D_":None, "E_":None}
            print(label)

        bboxes = o.predict(frame)
        if (bboxes):
            for bbox in bboxes:
                if (bbox[2] == 'person' and bbox[0][0] > 0 and bbox[0][1] > 0 and bbox[0][2] > 0 and bbox[0][3] > 0):
                    # person.append([int(((bbox[0][0] + bbox[0][2])*ratio_w)/2), int(((bbox[0][1] + bbox[0][3])*ratio_h)/2)])

                    person = insert_person(m, person, [bbox[0][0] * ratio_w, bbox[0][1] * ratio_h, bbox[0][2] * ratio_w, bbox[0][3] * ratio_h])

                    b = frame[int(bbox[0][1] * ratio_h): int(bbox[0][3] * ratio_h), int(bbox[0][0] * ratio_w): int(bbox[0][2] * ratio_w)]

                    frame = cv2.rectangle(frame, (int(bbox[0][0] * ratio_w), int(bbox[0][1] * ratio_h)), (int(bbox[0][2] * ratio_w), int(bbox[0][3] * ratio_h)), (255, 0, 0), 2)
                    # img = cv2.putText(img, motion, (int(bbox[0][0] * ratio_w), int(bbox[0][1] * ratio_h)),
                    #                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        print(person)

        cv2.imshow("a", frame)
        pre_label = label

        if cv2.waitKey(1) == ord('q'):
            break
        #cv2.waitKey(0)

def insert_person(m, person, bbox):
    index = ["A", "B", "C", "D", "E"]

    for i in index:
        if not (person[i]):
            person[i].append(bbox)
            break

        else:
            iou = cal_mean_iou(bbox, [person[i][-1]])
            if( 0.5  < iou ):
                person[i].append(bbox)
                if(len(person[i]) >= 3):
                    person[i+"_"] = m.test(person[i][-3:])
                break
            else:
                pass

    return person

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
motion_classify("_data/motion_test.mp4")

#count = 0
#for v in videos:
#    print(count)
#    count = get_motion_data(v, count, start=1000, interval=5)


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