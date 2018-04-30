import cv2
import csv

class Make_SceneData():
    def __init__(self):
        print("sceneData")

    def set_path(self, path):
        self.path = path

    def save_image_with_frame_interval(self, start=10000, end=130000, interval=5):
        video_path = "_data/" + self.path + "/" + self.path + ".mp4"
        path = "_data/" + self.path + "/"
        data = []
        video = cv2.VideoCapture(video_path)
        print("start")
        while True:
            video.set(1, start)
            success, frame = video.read()
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

    def change_9_to_13(self, p):
        path = "../_data/" + p + "/"
        csv_path = path + p + ".csv"
        csv_path2 = path + p + "2.csv"
        print(csv_path)

        f = open(csv_path, "r")
        f2 = open(csv_path2, "w", newline='')
        reader = csv.reader(f)

        sett = {"start": None, "end": None, "label": None}
        keys = ["start", "end", "label"]
        writer = csv.DictWriter(f2, keys)

        data = []
        for line in reader:
            if (int(line[0]) < int(line[1]) and int(line[1]) - int(line[0]) < 200):
                sett = {"start": line[0], "end": line[1], "label": line[2]}

                if(sett["label"] == '9' or sett["label"] == "7"):


                    new_data = []
                    for j in range(int(sett["start"]), int(sett["end"]) + 1):

                        cv2.imshow(
                            "a",
                            cv2.imread(
                                path+str(j)+".jpg"
                            )
                        )
                        key = cv2.waitKey(0)

                        if(str(key) == "97"): #a
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


                        new = {"image": str(j), "label": label}
                        new_data.append(new)
                        print(str(label)+"asdasdas"+str(j))



                    result_data = []
                    before = {"image": None, "label": None}
                    new_sett = {"start": None, "end": None, "label": None}

                    for i in new_data:
                        if(i["label"] != before["label"]):
                            if(before["label"] != None):
                                new_sett["end"] = before["image"]
                                print(new_sett)
                                result_data.append(new_sett)

                            new_sett = {"start": i["image"], "end": None, "label": i["label"]}


                        before = i

                    new_sett["end"] = before["image"]
                    result_data.append(new_sett)

                    data = data + result_data
                else:
                    data.append(sett)

        writer.writerows(data)



        f.close()
        f2.close()
