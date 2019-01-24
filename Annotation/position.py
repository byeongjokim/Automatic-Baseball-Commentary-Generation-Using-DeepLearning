import math

class Position(object):
    def __init__(self, motion, frame_shape, resource):
        self.motion = motion
        self.frame_shape = frame_shape
        self.resource = resource

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

    def print_bbox__(self):
        _ = [i+":"+str(self.person_bbox[i+"_"]) for i in self.field if(self.person_bbox[i+"_"])]
        print(_)

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

        self.pre_motions = [self.person_bbox[i+"_"] for i in self.field]

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
            self.new_motions = [self.person_bbox[i+"_"] for i in self.field]
            return 1

        self.new_motions = [self.person_bbox[i+"_"] for i in self.field]

    def is_motion_changed(self, i):
        index = self.field.index(i)
        if(self.new_motions[index] == self.pre_motions[index]):
            return 0
        else:
            return 1

    def annotation(self, label, person_bbox):
        if (label == 0):
            if (person_bbox["pitcher_"] == "pitching" and self.is_motion_changed("pitcher")):
                return self.pitcher_when_pitching(self.resource.get_pitcher())
            if (person_bbox["pitcher_"] == "pitching" and person_bbox["batter_"] == "batting" and self.is_motion_changed("batter")):
                return self.batter_when_batting(self.resource.get_batter())

        elif (label == 2):
            if (person_bbox["close_up_"]  and self.is_motion_changed("close_up")):
                m = person_bbox["close_up_"]
                #if (m == "run"):
                #    return "클로즈업 된 선수 뛰고 있습니다."
                #if (m == "walking"):
                #    return "클로즈업 된 선수 걷고 있습니다."


        elif (label == 5):  # first base
            if (person_bbox["1st_"] and self.is_motion_changed("1st")):
                m = person_bbox["1st_"]
                if (m == "catch_field"):
                    player = self.get_player_with_position("1st")
                    annotation = ["1루수 공을 잡았습니다.", player + "선수 공을 잡았습니다.", "1루수 " + player + "선수 공을 잡았습니다."]
                    return annotation

        elif (label == 8):  # second base
            if (person_bbox["2nd_"] and self.is_motion_changed("2nd")):
                m = person_bbox["2nd_"]
                if (m == "catch_field"):
                    player = self.get_player_with_position("2nd")
                    annotation = ["2루수 공을 잡았습니다.", player + "선수 공을 잡았습니다.", "2루수 " + player + "선수 공을 잡았습니다."]
                    return annotation

        elif (label == 10):  # third base
            if (person_bbox["3rd_"] and self.is_motion_changed("3rd")):
                m = person_bbox["3rd_"]
                if (m == "catch_field"):
                    player = self.get_player_with_position("3rd")
                    annotation = ["3루수 공을 잡았습니다.", player + "선수 공을 잡았습니다.", "3루수 " + player + "선수 공을 잡았습니다."]
                    return annotation

        elif (label == 6):  # center outfield
            if (person_bbox["center fielder_"] and self.is_motion_changed("center fielder")):
                m = person_bbox["center fielder_"]
                player = self.get_player_with_position("COF")
                return self.field_motion("중견수", player, m)

        elif (label == 7):  # right outfield
            if (person_bbox["right fielder_"] and self.is_motion_changed("right fielder")):
                m = person_bbox["right fielder_"]
                player = self.get_player_with_position("ROF")
                return self.field_motion("우익수", player, m)

        elif (label == 11):  # left outfield
            if (person_bbox["left fielder_"] and self.is_motion_changed("left fielder")):
                m = person_bbox["left fielder_"]
                player = self.get_player_with_position("LOF")
                return self.field_motion("좌익수", player, m)

        elif (label == 12):  # ss
            if (person_bbox["ss_"] and self.is_motion_changed("ss")):
                m = person_bbox["ss_"]
                player = self.get_player_with_position("ss")
                return self.field_motion("유격수", player, m)

        return None

    def get_player_with_position(self, position):
        btop = self.resource.get_btop()
        if btop == 0:
            lineup = self.resource.get_LineUp(0)
        else:
            lineup = self.resource.get_LineUp(1)

        return self.get_player_name(lineup[position]["name"])

    def pitcher_when_pitching(self, pitcher):
        pitcher = self.get_player_name(pitcher)
        annotation = [
            str(pitcher) + " 투수 공을 던졌습니다.",
            "공을 던졌습니다.",
            str(pitcher) + " 투수 타자를 향해 힘껏 공을 던졌습니다.",
        ]

        return annotation

    def batter_when_batting(self, batter):
        batter = self.get_player_name(batter)
        annotation = [
            str(batter) + "타자 배트를 휘둘렀습니다.",
            str(batter) + "타자 힘차게 배트를 휘둘렀습니다.",
            "배트를 휘둘렀습니다.",
            "스윙!",
        ]

        return annotation


    def field_motion(self, position, player, motion):
        player = self.get_player_name(player)
        annotation = None
        if(motion  == "throwing"):
            annotation = [position + " 송구 하였습니다.", player + "선수 송구 하였습니다.", position + " " + player + "선수 송구 하였습니다."]
        elif(motion == "catch_field"):
            annotation = [position + " 공을 잡았습니다.", player + "선수 공을 잡았습니다.", position + " " + player + "선수 공을 잡았습니다."]
        elif(motion == "run" or motion == "walking"):
            annotation = [position + " 쪽 입니다.", player + "선수 쪽 입니다.", position + " " + player + "선수 쪽 입니다.", position + " " + player + "선수"]
        return annotation

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

    def get_player_name(self, name):
        return "".join([s for s in list(name) if not s.isdigit()])