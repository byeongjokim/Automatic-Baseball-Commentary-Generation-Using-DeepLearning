# -*- coding: utf-8 -*-
import cv2
from NN.scene_model import Scene_Model


class SceneData():

    prev = -1

    def __init__(self, Resources, shape=(320, 180)):
        print("init_sceneData")
        self.Resources = Resources

        self.width = shape[0]
        self.height = shape[1]

        self.scene = Scene_Model()
        self.scene.make_model()


    def Annotation_with_frame(self, frame, relayText):
        #print("\t\t\t\t대기시간이 길어 영상처리로 텍스트 생성")

        label = self.scene.predict(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (self.width, self.height))

        """
        people, full = self.get_human(resize)
        for (x, y, w, h) in people:
            person = resize[y:y + h, x:x + w]
            person_resize = cv2.resize(person, (self.Resources.motion_weight, self.Resources.motion_height))
            person_image = np.array(person_resize)
            #motion = self.predict_motion(person_image, full)

            cv2.rectangle(resize, (x, y), (x + w, y + h), (0, 0, 255), 2)
        """

        #return label

        if(self.prev != label):

            if (label == 0):
                print("\t\t투수, 타자 그리고 포수가 영상에 잡히네요.")

            elif(label == 1):
                #print("\t\t"+str(relayText["batorder"])+"번 타자의 모습입니다.")
                print("\t\tn번 타자의 모습입니다.")

            elif (label == 2):
                print("\t\t선수들이 클로즈업 되었네요. 혹은 투수")

            elif (label == 3):
                print("\t\t코치들의 모습이네요.")

            elif (label == 4):
                print("\t\t관중들이 응원을 하고 있습니다.")

            elif (label == 5):
                print("\t\t1루쪽 내야 입니다.")

            elif (label == 6):
                print("\t\t경기장 외야 입니다.")

            elif (label == 7):
                print('\t\t1루쪽 외야 입니다.')

            elif (label == 8):
                print("\t\t경기장 내야 입니다.")

            elif (label == 9):
                print("\t\t기타 장면 입니다..")

            elif (label == 10):
                print("\t\t3루쪽 내야 입니다.")

            elif (label == 11):
                print("\t\t3루쪽 외야 입니다.")

            print("\t\t============================")

            self.prev = label

        return label

