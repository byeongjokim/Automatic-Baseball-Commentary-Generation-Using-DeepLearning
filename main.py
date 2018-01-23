import threading
import cv2
from Video.Video import Video
from Annotation.Annotation import Annotation
from Person.Person import Person
from resources import Resources
import numpy as np
from Annotation.Scene_data import Make_SceneData
from imutils.object_detection import non_max_suppression

'''
s = Make_SceneData('./_data/scene1-1.csv')
s.clustering()
s.save_image_data()
'''

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
        person_resize = cv2.resize(person, (60, 80))
        person_image = np.array(person_resize)

        text = act.predict(person_image)
        cv2.imwrite(str(count)+".jpg", person_image)
        count = count+1


#train_act()
#test_act("./_data/scene_image/field_48.jpg")


resources = Resources()

annotation = Annotation('./_data/20171030KIADUSAN.txt', resources)
person = Person(annotation, resources)
video = Video(resources)

o_start = "183122"
o_count = 8145
fps = 29.97

#count = 70233  before start 2
count = 10000

naver = threading.Thread(target=annotation.generate_Naver, args=(count-o_count, fps, o_start, ))
naver.start()

scene = threading.Thread(target=annotation.generate_Scene)
scene.start()

video.play(v="./_data/20171030KIADUSAN.mp4", count=count)
