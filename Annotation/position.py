import math

class Position(object):
    def __init__(self, motion, frame_shape):
        self.motion = motion
        self.frame_shape = frame_shape

        self.count = 0

        self.field = ["1st", "2nd", "3rd", "ss", "right fielder", "left fielder", "center fielder", "pitcher", "catcher", "batter", "1st_runner", "2nd_runner", "3rd_runner", "close_up"]
        self.tmp_index = ["A", "B", "C", "D", "E", "F", "G"]

        self.person_bbox = {}
        self.person_image = {}

        for i in self.field:
            self.person_bbox[i] = []
            self.person_bbox[i + "_"] = None
            self.person_image[i] = []

        for i in self.tmp_index:
            self.person_bbox[i] = []
            self.person_bbox[i + "_"] = None
            self.person_image[i] = []

    def get_bbox(self):
        return self.person_bbox

    def clear(self):
        self.count = 0
        for i in self.field:
            self.person_bbox[i] = []
            self.person_bbox[i + "_"] = None
            self.person_image[i] = []

        for i in self.tmp_index:
            self.person_bbox[i] = []
            self.person_bbox[i + "_"] = None
            self.person_image[i] = []

    def insert_person(self, frame, bboxes, label):
        if(label == 0): #pitching batting
            for bbox in bboxes:
                for i in ["pitcher", "catcher", "batter"]:
                    if(bbox[4] == i):
                        self.to_pos(frame, bbox, i)

        elif(label == 1): #batter close up
            self.to_pos(frame, self.find_big_box(bboxes), "batter")

        elif(label == 2): #close up
            self.to_pos(frame, self.find_big_box(bboxes), "close_up")

        elif(label == 5): #first base
            self.to_pos(frame, self.find_center_box(bboxes), "1st")

        elif(label == 6): #center outfield
            self.to_pos(frame, self.find_center_box(bboxes), "center fielder")

        elif(label == 7): #right outfield
            self.to_pos(frame, self.find_center_box(bboxes), "right fielder")

        elif(label == 8): #second base
            self.to_pos(frame, self.find_center_box(bboxes), "2nd")

        elif(label == 10): #third base
            self.to_pos(frame, self.find_center_box(bboxes), "3rd")

        elif(label == 11): #left outfield
            self.to_pos(frame, self.find_center_box(bboxes), "left fielder")

        elif(label == 12): #ss
            self.to_pos(frame, self.find_center_box(bboxes), "ss")

        else:  # coach and gallery and etc
            return 1

    def annotation(self, label, person_bbox):
        if (label == 0):
            if (person_bbox["pitcher_"] == "pitching"):
                print("투수 공을 던졌습니다.")
            if (person_bbox["batter_"] == "batting"):
                print("타자 휘둘었습니다.")

        elif (label == 2):
            if (person_bbox["close_up_"]):
                m = person_bbox["close_up_"]
                if (m == "run"):
                    print("클로즈업 된 선수 뛰고 있습니다.")
                if (m == "walking"):
                    print("클로즈업 된 선수 걷고 있습니다.")

        elif (label == 5):  # first base
            if (person_bbox["1st_"]):
                m = person_bbox["1st_"]
                if (m == "catch_field"):
                    print("1루수 공을 잡았습니다.")
            return 1

        elif (label == 6):  # center outfield
            if (person_bbox["center fielder_"]):
                m = person_bbox["center fielder_"]
                print("중견수 " + m + " 하고있습니다.")
            return 1

        elif (label == 7):  # right outfield
            if (person_bbox["right fielder_"]):
                m = person_bbox["right fielder_"]
                print("우익수 " + m + " 하고있습니다.")
            return 1

        elif (label == 8):  # second base
            if (person_bbox["2nd_"]):
                m = person_bbox["2nd_"]
                if (m == "catch_field"):
                    print("2루수 공을 잡았습니다.")
            return 1

        elif (label == 10):  # third base
            if (person_bbox["3rd_"]):
                m = person_bbox["3rd_"]
                if (m == "catch_field"):
                    print("3루수 공을 잡았습니다.")
            return 1

        elif (label == 11):  # left outfield
            if (person_bbox["left fielder_"]):
                m = person_bbox["left fielder_"]
                print("좌익수 " + m + " 하고있습니다.")
            return 1

        elif (label == 12):  # ss
            if (person_bbox["ss_"]):
                m = person_bbox["ss_"]
                print("유격수 " + m + " 하고있습니다.")
            return 1

        else:  # coach and gallery and etc batter
            return 1

        return 1


    @staticmethod
    def cal_mean_iou(bbox1, bboxes2):
        s = 0.0
        for bbox2 in bboxes2:
            min_x = max(bbox1[0], bbox2[0])
            max_x = min(bbox1[2], bbox2[2])
            min_y = max(bbox1[1], bbox2[1])
            max_y = min(bbox1[3], bbox2[3])

            if (max_x < min_x or max_y < min_y):
                s = s + 0.0
            else:
                inter = (max_x - min_x) * (max_y - min_y)
                bb1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                bb2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

                iou = inter / (bb1 + bb2 - inter)
                if (iou >= 0.0 and iou <= 1.0):
                    s = s + iou
                else:
                    s = s + 0.0
        return s / len(bboxes2)

    """
    @staticmethod
    def find_8direction_from_box(bbox, bboxes):
        N = None
        NE = None
        E = None
        SE = None
        S = None
        SW = None
        W = None
        NW = None

        N = sorted(bboxes, key=lambda x:x[1])

        return [N, NE, E, SE, S, SW, W, NW]
    """

    @staticmethod
    def find_top_left_bottom_right_box(bboxes):
        top = sorted(bboxes, key=lambda x: x[1])[0]
        bottom = sorted(bboxes, key=lambda x: x[3], reverse=True)[0]
        left = sorted(bboxes, key=lambda x: x[0])[0]
        right = sorted(bboxes, key=lambda x: x[2], reverse=True)[0]
        return top, bottom, left, right

    @staticmethod
    def find_big_box(bboxes):
        return sorted(bboxes, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]), reverse=True)[0]

    def find_center_box(self, bboxes):
        return sorted(bboxes, key=lambda x: self.find_distance((self.frame_shape[0]/2, self.frame_shape[1]/2), self.find_center_point(x)))[0]

    @staticmethod
    def find_distance(pointA, pointB):
        return math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)

    @staticmethod
    def find_center_point(bbox):
        return ((bbox[1] + bbox[3])/2, (bbox[0] + bbox[2])/2)

    def to_pos(self, frame, bbox, i):
        p = frame[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]
        if not (self.person_bbox[i]):
            self.person_bbox[i].append(bbox)
            self.person_image[i].append(p)
            self.person_bbox[i + "_"] = self.motion.test(self.person_image[i])
            return 1

        else:
            iou = self.cal_mean_iou(bbox, [self.person_bbox[i][-1]])
            if (0.5 < iou):
                self.person_bbox[i].append(bbox)
                self.person_image[i].append(p)
                self.person_bbox[i + "_"] = self.motion.test(self.person_image[i])
                return 1
            else:
                return 0
