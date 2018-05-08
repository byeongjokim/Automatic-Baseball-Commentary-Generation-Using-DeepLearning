import cv2
import csv
import os

class Make_SceneData():
    def __init__(self):
        print("sceneData")

    def set_path(self, path):
        self.path = path
        os.mkdir("_data/"+path)
        print("mkdir _data/"+path)

    def set_video(self, v):
        self.video  = cv2.VideoCapture(v)

    def save_image_with_frame_interval(self, start=10000, end=130000, interval=5):
        #video_path = "_data/" + self.path + "/" + self.path + ".mp4"
        path = "_data/" + self.path + "/"
        data = []
        #video = cv2.VideoCapture(video_path)
        print("start")
        while True:
            self.video.set(1, start)
            success, frame = self.video.read()
            if not success:
                break
            if (start > end):
                break

            data.append(frame)
            start = start + interval
        print("end")
        print("save start")
        count = 0
        for i in data:
            cv2.imwrite(path + str(count) + ".jpg", i)
            count = count + 1

    def labeling(self, p):
        path = "../_data/" + p + "/"
        csv_path2 = path + p + ".csv"

        f2 = open(csv_path2, "w", newline='')

        sett = {"start": None, "end": None, "label": None}
        keys = ["start", "end", "label"]
        writer = csv.DictWriter(f2, keys)

        data = []
        new_data = []
        for j in range(0, 2800):

            cv2.imshow(
                "a",
                cv2.imread(
                    path+str(j)+".jpg"
                )
            )
            key = cv2.waitKey(0)


            if (str(key) == "48"):  # 0 etc
                label = "10"
            elif (str(key) == "49"):  # 1 pitchingbatting
                label = "1"
            elif (str(key) == "50"):  # 2 batter
                label = "2"
            elif (str(key) == "51"):  # 3 closeup
                label = "3"
            elif (str(key) == "52"):  # 4 coach
                label = "4"
            elif (str(key) == "53"):  # 5 gallery
                label = "5"

            elif(str(key) == "97"): #a
                label = "11"
            elif(str(key) == "115"): #s
                label = "13"
            elif(str(key) == "100"): #d
                label = "9"
            elif (str(key) == "102"): #f
                label = "6"
            elif (str(key) == "113"): #q
                label = "12"
            elif (str(key) == "101" or str(key) == "119"):  # e, w
                label = '7'
            elif (str(key) == "114"): #r
                label = "8"

            elif (str(key) == "125"):
                break

            else:
                label = None


            if(label):
                new = {"image": str(j), "label": label}
                new_data.append(new)
                print(str(label)+"asdasdas"+str(j))




        result_data = []
        before = {"image": None, "label": None}
        new_sett = {"start": None, "end": None, "label": None}

        for i in new_data:
            if(i["label"] != before["label"] or int(i["image"])-int(before["image"]) != 1 ):
                if(before["label"] != None):
                    new_sett["end"] = str(int(before["image"])-1)
                    result_data.append(new_sett)

                new_sett = {"start": i["image"], "end": None, "label": i["label"]}


            before = i

        new_sett["end"] = str(int(before["image"])-1)
        if(int(new_sett["end"]) > int(new_sett["start"])):
            result_data.append(new_sett)
            print(result_data)
        data = data + result_data

        writer.writerows(data)


        f2.close()

a = Make_SceneData()
a.labeling("180404SAMNC")