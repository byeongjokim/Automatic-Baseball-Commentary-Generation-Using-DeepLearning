import json
import random
from Annotation.String import *

class PitchingBatting():
    def set(self, batterbox_no, relayText, ball_data):
        text = relayText["liveText"]

        inn = relayText["inn"]
        btop = relayText["btop"]
        pitchId = relayText["pitchId"]

        ballcount = relayText["ballcount"]

        batter = ball_data["batterName"]
        batorder = relayText["batorder"]
        ilsun = relayText["ilsun"]

        pitcher = ball_data["pitcherName"]
        stuff = ball_data["stuff"]


        annotation = ""
        if ("구 스트라이크" in text):
            annotation = self.Strike(ball_data)

        elif("구 헛스윙" in text):
            annotation = self.Swing(ball_data)

        elif ("구 볼" in text):
            annotation = self.Ball(ball_data)

        elif ("구 파울" in text):
            annotation = self.Foul(ball_data)

        elif ("구 번트파울" in text):
            ballCount = str(ball_data["ballcount"])
            Speed = str(ball_data["speed"])
            stuff = str(ball_data["stuff"])

            annotation = buntFoul(ballCount, Speed, stuff)

        elif ("구 타격" in text):
            ballCount = str(ball_data["ballcount"])
            Speed = str(ball_data["speed"])
            stuff = str(ball_data["stuff"])

            annotation = hit(ballCount, Speed, stuff)

        print(annotation)


    def Strike(self, ball_data):
        ball_loc = self.calc_ball_location(ball_data)

        ballCount = str(ball_data["ballcount"])
        Speed = str(ball_data["speed"])
        stuff = str(ball_data["stuff"])

        if("ID" == ball_loc or "IU" == ball_loc or "OD" == ball_loc or "OU" == ball_loc):
            return difficultCourseStrike(ballCount, Speed, stuff)

        elif("MM" == ball_loc):
            return easyCourseStrike(ballCount, Speed, stuff)

        elif("IM" == ball_loc):
            return insideStrike(ballCount, Speed, stuff)

        elif("OM" == ball_loc):
            return outsideStrike(ballCount, Speed, stuff)

        elif("MD" == ball_loc):
            return downsideStrike(ballCount, Speed, stuff)

        elif("MU" == ball_loc):
            return upsideStrike(ballCount, Speed, stuff)

    def Ball(self, ball_data):
        ball_loc = self.calc_ball_location(ball_data)

        ballCount = str(ball_data["ballcount"])
        Speed = str(ball_data["speed"])
        stuff = str(ball_data["stuff"])

        if ("ID" == ball_loc or "IU" == ball_loc or "OD" == ball_loc or "OU" == ball_loc):
            return difficultCourseBall(ballCount, Speed, stuff)

        elif ("MM" == ball_loc):
            return easyCourseBall(ballCount, Speed, stuff)

        elif ("IM" == ball_loc):
            return insideBall(ballCount, Speed, stuff)

        elif ("OM" == ball_loc):
            return outsideBall(ballCount, Speed, stuff)

        elif ("MD" == ball_loc):
            return downsideBall(ballCount, Speed, stuff)

        elif ("MU" == ball_loc):
            return upsideBall(ballCount, Speed, stuff)

    def Swing(self, ball_data):
        ball_loc = self.calc_ball_location(ball_data)

        ballCount = str(ball_data["ballcount"])
        Speed = str(ball_data["speed"])
        stuff = str(ball_data["stuff"])

        if ("ID" == ball_loc or "IU" == ball_loc or "OD" == ball_loc or "OU" == ball_loc):
            return difficultCourseSwing(ballCount, Speed, stuff)

        elif ("MM" == ball_loc):
            return easyCourseSwing(ballCount, Speed, stuff)

        elif ("IM" == ball_loc):
            return insideSwing(ballCount, Speed, stuff)

        elif ("OM" == ball_loc):
            return outsideSwing(ballCount, Speed, stuff)

        elif ("MD" == ball_loc):
            return downsideSwing(ballCount, Speed, stuff)

        elif ("MU" == ball_loc):
            return upsideSwing(ballCount, Speed, stuff)

    def Foul(self, ball_data):
        ballCount = str(ball_data["ballcount"])
        Speed = str(ball_data["speed"])
        stuff = str(ball_data["stuff"])

        if(int(ballCount) >= 6):
            return cut(ballCount, Speed, stuff)
        else:
            return normalFoul(ballCount, Speed, stuff)

    def calc_ball_location(self, ball):
        H = ""
        V = ""

        if (float(ball["crossPlateX"]) > 0.7):
            if (ball["stance"] == "R"):
                H = "I"
            else:
                H = "O"
        elif (float(ball["crossPlateX"]) < -0.7):
            if (ball["stance"] == "L"):
                H = "I"
            else:
                H = "O"
        else:
            H = "M"

        t = 18.4 * 3.6 / float(ball["speed"]) + 0.15
        z = (float(ball["az"]) * t * t / 2) + (float(ball["vz0"]) * t + float(ball["z0"]))

        if (z > 1.2):
            V = "U"
        elif (z < -1.2):
            V = "D"
        else:
            V = "M"

        return H + V


class GoBase():
    def __init__(self):
        print("go base class")

    def set(self, relayText):
        text = relayText["liveText"]

        if("진루" in text):
            return 1
        elif("출루" in text):
            return 1
        elif("홈인" in text):
            return 1

class Out():
    def __init__(self):
        print("Out class")

    def set(self, relayText):
        text = relayText["liveText"]

        if("삼진" in text):
            return 1
        else:
            return 1
class Etc():
    def __init__(self):
        print('Etc class')

    def change(self):
        return 1

    def steal(self):
        return 1

    def inning_start(self):
        return 1