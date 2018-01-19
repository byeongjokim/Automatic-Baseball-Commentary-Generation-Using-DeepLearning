# -*- coding: utf-8 -*-
import csv
import cv2
import os
import numpy as np
from skimage.measure import compare_ssim as ssim

class SceneData():
    def __init__(self, Resources, shape=(320,180)):
        print("init_sceneData")
        self.Resources = Resources

        self.width = shape[0]
        self.height = shape[1]

        self.load_image_data()

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

    def compare_images(self, A, B):
        s = ssim(A, B, multichannel=True)
        #print("MSE: %.2f, struct_SSIM: %.2f" % (m, s))
        return s

    def predict(self, frame_no, relayText):
        #print("\t\t\t\t대기시간이 길어 영상처리로 텍스트 생성")

        video = cv2.VideoCapture("./_data/20171030KIADUSAN.mp4")

        video.set(1, frame_no)
        success, frame = video.read()

        if not success:
            print("can not load video")
            return 0

        self.predict_motion(frame)

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
            print("\t\t\t\t"+str(relayText["batorder"])+"번 타자의 모습입니다. -> 추후 선수 정보")
        elif (label == "pitchingbatting"):
            print("\t\t\t\t투수, 타자 그리고 포수가 영상에 잡히네요. 어떤 공을 던질까요?")
        elif (label == "pitcher"):
            print("\t\t\t\t투수의 모습입니다. -> 추후 선수 정보")
        elif (label == "run"):
            print("\t\t\t\t뛰고 있네요.")
        elif (label == "coach"):
            print("\t\t\t\t코치들의 모습이네요.")
        else:
            print('\t\t\t\t기타 장면 입니다.')

        cv2.imwrite(str(frame_no)+".jpg", frame)

    def predict_motion(self, frame):
        print("predict motion")

    def predict2(self, frame, relayText):
        #print("\t\t\t\t대기시간이 길어 영상처리로 텍스트 생성")

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
            print("\t\t\t\t"+str(relayText["batorder"])+"번 타자의 모습입니다. -> 추후 선수 정보")
        elif (label == "pitchingbatting"):
            print("\t\t\t\t투수, 타자 그리고 포수가 영상에 잡히네요. 어떤 공을 던질까요?")
        elif (label == "pitcher"):
            print("\t\t\t\t투수의 모습입니다. -> 추후 선수 정보")
        elif (label == "run"):
            print("\t\t\t\t뛰고 있네요.")
        elif (label == "coach"):
            print("\t\t\t\t코치들의 모습이네요.")
        else:
            print('\t\t\t\t기타 장면 입니다.')

class Make_SceneData():
    def __init__(self, path, shape=(320,180),fps=29.970):
        print("sceneData")
        self.path = path
        self.fps = fps
        self.get_data_csv()

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
                else:
                    result.append({"SceneNumber": count, "start": int(line[1]), "end": int(float(line[4]) * self.fps) + int(line[1]), "label": None})
                count = count + 1

        self.data = result[:354]
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

            #cv2.imwrite(str(i["SceneNumber"])+".jpg", frame)
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


