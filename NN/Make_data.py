import cv2

class Make_SceneData():
    def __init__(self, video, fps=29.970):
        print("sceneData")
        self.video = video
        self.fps = fps

    def save_image_with_frame_interval(self, start=60000, end=130000, interval=5):

        data = []
        video = cv2.VideoCapture(self.video)
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
            print(count)
            cv2.imwrite("./motion_data/20171029KIADUSAN/" + str(count) + ".jpg", i)
            count = count + 1

    def amplification(self):

        """
        train_data = []
        test_data = []

        path = "./scene_data/train/"
        image = []
        for (p, dir, files) in os.walk(path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.jpg':
                    image.append(filename)

        train_data=[]
        for i in image:
            train_data.append({"image":
                cv2.cvtColor(
                    cv2.imread(path + i),
                    cv2.COLOR_BGR2GRAY),
                "label": i.split(".")[0].split("_")[0]})


        video = cv2.VideoCapture("./_data/20171030KIADUSAN.mp4")

        for i in self.data:
            no_frame = (i["start"] + i["end"]) / 2
            video.set(1, no_frame)
            success, frame = video.read()

            if not success:
                break

            test_data.append({"image":frame, "label":None})

        print("made %d test, %d train data" %(len(test_data), len(train_data)))
        print("will calculate simm")

        count = 0
        for i in test_data:
            print(count)

            result = []
            for j in train_data:
                s = self.compare_images(j["image"], cv2.cvtColor(i["image"], cv2.COLOR_BGR2GRAY))

                result.append({"label":j["label"], "ssim":s})

            result.sort(key=operator.itemgetter('ssim'), reverse=True)
            print(result)
            result = result[:3]
            print(result)
            l = [i["label"] for i in result]

            first = result[0]["label"]

            counter = Counter(l)
            print(counter)
            if(counter.most_common()[0][1] == 1):
                label = first
            else:
                label = counter.most_common()[0][0]

            print(label)
            i["label"] = label
            cv2.imwrite("./scene_data/test/"+str(i["label"])+"_"+str(count)+".jpg", i["image"])
            count = count + 1
        #self.result = test_data + train_data
        """
        return 1
