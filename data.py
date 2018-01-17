# -*- coding: utf-8 -*-
import json
import operator
import time
import csv
import cv2
import os
import threading
import tensorflow as tf
import numpy as np
#from util import ssim, msssim
from skimage.measure import compare_ssim as ssim
import random


from NN.cnn import conv_layer, pool

class TextData():
    frameNo = 0
    threshold = 5

    def __init__(self, filename='./_data/20171030KIADUSAN.txt'):
        data_file = open(filename, "rt", encoding="UTF8")
        data = json.load(data_file)
        data_file.close()
        self.game_info = data["gameInfo"]
        self.relayTexts = self.sort_with_seqno(data["relayTexts"])

        self.sceneData = SceneData("./_data/scene1-1.csv")
        self.sceneData.load_image_data()


    def sort_with_seqno(self, relayTexts):
        newlist = []
        newlist = newlist + relayTexts['1']
        newlist = newlist + relayTexts['2']
        newlist = newlist + relayTexts['3']
        newlist = newlist + relayTexts['4']
        newlist = newlist + relayTexts['5']
        newlist = newlist + relayTexts['6']
        newlist = newlist + relayTexts['7']
        newlist = newlist + relayTexts['8']
        newlist = newlist + relayTexts['9']
        newlist = newlist + relayTexts['currentBatterTexts']
        newlist = newlist + [relayTexts['currentOffensiveTeam']]
        newlist = newlist + [relayTexts['currentBatter']]

        newlist.sort(key=operator.itemgetter("seqno"))
        #for i in newlist:
        #    print(i)

        return newlist

    def get_timedelta(self, end, start):
        h1 = start[:2]
        m1 = start[2:4]
        s1 = start[4:]

        s = int(h1)*12 + int(m1)*60 + int(s1)

        h2 = end[:2]
        m2 = end[2:4]
        s2 = end[4:]

        e = int(h2) * 12 + int(m2) * 60 + int(s2)

        return e-s

    def setframeNo(self, frameNo):
        self.frameNo = frameNo
        #print(self.frameNo)

    def return_seq(self):

        start = "183122"
        count = 0
        pre_pitchId = '000000'
        tmp_pitchId = '000000'

        for i in self.relayTexts:
            wait = 0
            if(count == 0):
                now = "183122"

            else:
                pre_pitchId = now
                now = i["pitchId"].split("_")[-1]

                if(now != '-1'):
                    if(pre_pitchId == '-1'):
                        wait = self.get_timedelta(now, tmp_pitchId)

                    else:
                        wait = self.get_timedelta(now, pre_pitchId)

                    #print("asd" + str(self.frameNo))

                    if (wait > self.threshold):
                        time.sleep(self.threshold)
                        t = threading.Thread(target=self.sceneData.predict, args=(self.frameNo,))
                        t.start()
                        time.sleep(wait-self.threshold)
                    else:
                        time.sleep(wait)

                else: #now == -1
                    if(pre_pitchId != '-1'):
                        tmp_pitchId = pre_pitchId
                    wait = 0

            print(i["liveText"])

            count = count + 1



class SceneData():
    def __init__(self, path, shape=(320,180),fps=29.970):
        print("sceneData")
        self.path = path
        self.fps = fps
        #self.get_data_csv()

        self.width = shape[0]
        self.height = shape[1]

    def get_data_csv(self):
        f = open(self.path, 'r', encoding='utf-8')
        reader = csv.reader(f)
        result = []
        count = 1
        for line in reader:
            if not line:
                pass

            elif(line[0] == str(count)):
                if(line[5]):
                    result.append({"SceneNumber":count, "start":int(line[1]), "end":int(float(line[4])*self.fps) + int(line[1]), "label":line[5]})
                count = count + 1

        self.data = result
        f.close()

    def save_image_data(self):
        video = cv2.VideoCapture("./_data/20171030KIADUSAN.mp4")

        count = 0
        for i in self.data:
            no_frame = (i["start"] + i["end"]) / 2
            video.set(1, no_frame)
            success, frame = video.read()

            if not success:
                break

            cv2.imwrite(i["label"]+"_"+str(count)+".jpg", frame)
            count = count + 1

        return 1

    def load_image_data(self):
        path = "./_data/scene_image/"
        image = []
        for (p, dir, files) in os.walk(path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.jpg':
                    image.append(filename)

        self.image_data = []
        for i in image:
            self.image_data.append({"image":cv2.resize(
                                            cv2.cvtColor(
                                                cv2.imread(path+i),
                                                cv2.COLOR_BGR2GRAY),
                                            (self.width, self.height)),
                                    "label":i.split(".")[0].split("_")[0]})

        print("we have %d image data" %(len(self.image_data)))


    def predict(self, frame_no):
        print("\t\t\t\t대기시간이 길어 영상처리로 텍스트 생성")
        image = []

        video = cv2.VideoCapture("./_data/20171030KIADUSAN.mp4")

        video.set(1, frame_no)
        success, frame = video.read()

        if not success:
            print("can not load video")
            return 0

        cv2.imwrite(str(frame_no)+".jpg", frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (self.width, self.height))

        result = []
        for i in self.image_data:
            result.append(self.compare_images(i["image"], resize))

        label = self.image_data[result.index(max(result))]["label"]

        if(label == "beforestart"):
            print("\t\t\t\t경기 시작 전입니다.")
        elif(label == "field"):
            print("\t\t\t\t경기장을 보여주고 있습니다.")
        elif (label == "gallery"):
            print("\t\t\t\t관중들이 응원을 하고 있습니다.")
        elif (label == "closeup"):
            print("\t\t\t\t선수들이 클로즈업 되었네요. -> 추후 선수정보")
        elif (label == "practice"):
            print("\t\t\t\t투수가 연습 구를 던지고 있습니다.")
        elif (label == "batter"):
            print("\t\t\t\t타자의 모습입니다. -> 추후 선수 정보")
        elif (label == "pitchingbatting"):
            print("\t\t\t\t투수, 타자 그리고 포수가 영상에 잡히네요. 어떤 공을 던질까요?")
        elif (label == "pitcher"):
            print("\t\t\t\t투수의 모습입니다. -> 추후 선수 정보")
        elif (label == "run"):
            print("\t\t\t\t뛰고 있네요.")
        else:
            print('기타 장면 입니다.')

    def mse(self, A, B):
        err = np.sum((A.astype('float') - B.astype('float')) ** 2)
        err /= float(A.shape[0] * A.shape[1])
        return err

    def compare_images(self, A, B):
        m = self.mse(A, B)
        s = ssim(A, B, multichannel=True)
        #print("MSE: %.2f, struct_SSIM: %.2f" % (m, s))
        return s


    def clustering(self):
        train_data = []
        test_data = []
        video = cv2.VideoCapture("./_data/20171030KIADUSAN.mp4")

        for i in self.data:
            no_frame = (i["start"] + i["end"]) / 2
            video.set(1, no_frame)
            success, frame = video.read()

            if not success:
                break

            cv2.imwrite(str(i["SceneNumber"])+".jpg", frame)
            if(i["label"]):
                test_data.append({"image":cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY), "no":i["SceneNumber"], "label":i["label"]})
            else:
                train_data.append({"image":cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY), "no":i["SceneNumber"], "label":None})

        print("made %d test, %d train data" %(len(test_data), len(train_data)))
        print("will calculate simm")

        count = 0
        for i in train_data:
            print(count)
            result = []
            for j in test_data:
                s = self.compare_images(j["image"], i["image"])
                result.append(s)

            max_id = result.index(max(result))
            self.data[i["no"]-1]["label"] = test_data[max_id]["label"]
            count = count + 1

        for i in self.data:
            print(i)

        with open("result.csv", "wb") as csv_file:
            writer = csv.writer(csv_file)
            for i in self.data:
                for key, value in i.items():
                    writer.writerow([key, value])
        #cv2.imwrite(str(i["no"])+".jpg", i["image"])

