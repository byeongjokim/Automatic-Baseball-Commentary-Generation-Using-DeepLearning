import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from NN.scene_model import Make_SceneData, Scene_Model
from NN.motion_model import Data, Action

def train_act():
    size = [60, 80]
    data = Data(size=size)
    train_x, train_y, test_x, test_y, actions = data.make_train_data()
    print("made "+str(len(train_x))+" train_data")
    #print("made "+str(len(test_x))+" test_data")

    act = Action(actions, size=size)
    act.make_model()
    act.train(train_x, train_y)
    act.test(test_x, test_y)

def test_act(img):

    body_cascade = cv2.CascadeClassifier('./_data/cascades/haarcascade_fullbody.xml')
    #body_cascade = cv2.CascadeClassifier('./_data/cascades/haarcascade_upperbody.xml')

    act = Action(4, size=[60, 80])
    act.make_model()

    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    people = body_cascade.detectMultiScale(gray, 1.05, 3, flags=cv2.CASCADE_SCALE_IMAGE)
    people = non_max_suppression(people, probs=None, overlapThresh=0.75)

    count = 0
    for (x, y, w, h) in people:
        person = gray[y:y + h, x:x + w]
        #person = cv2.resize(person, (60, 80))
        person_image = np.array(person)

        #text = act.predict(person_image)
        cv2.imwrite(str(count)+".jpg", person_image)
        count = count+1

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    (rects, weights) = hog.detectMultiScale(gray, winStride=(16,16), padding=(8,8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.75)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.imwrite("asd"+".jpg", image)

def train_scene():
    s = Scene_Model()
    s.load_data()

    s.make_model()
    s.train()
    s.test()

def test_scene():
    s = Scene_Model()
    s.num_label = 10
    image = cv2.imread("./scene_data/train/pitchingbatting_230.jpg")
    s.make_model()
    s.predict(image)

#train_act()
#test_act("./_data/scene_image/pitcher_39.jpg")
#train_scene()
test_scene()
