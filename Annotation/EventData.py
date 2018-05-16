import json
import random
from Annotation.String import *

class Change():
    def __init__(self, GameInfo, TeamLineup):
        self.GameInfo = GameInfo
        self.TeamLineup = TeamLineup

    def set(self, relayText):
        text = relayText["liveText"]

        annotation = ""
        if("번타자" in text):
            annotation = self.batterBox(relayText)

        elif("교체" in text):
            annotation = self.change_player(relayText)

        elif("변경" in text):
            annotation = self.change_position(relayText)

        elif("종료" in text):
            annotation = self.fin_attack(relayText)

        print(annotation)

    def batterBox(self, relayText):
        batorder = relayText["batorder"]
        btop = relayText["btop"]

        batter = self.get_batter(batorder, btop)[0]

        return batter_Start(batter)

    def change_player(self, relayText):
        return relayText["liveText"]

    def change_position(self, relayText):
        return relayText["liveText"]

    def fin_attack(self, relayText):
        inn = relayText["inn"]
        btop = relayText["btop"]
        score = {"homeScore" : relayText["homeScore"], "awayScore" : relayText["awayScore"]}
        innscore = {"homeInningScore" : relayText["homeInningScore"], "awayInningScore" : relayText["awayInningScore"]}
        return inn_end(self.GameInfo, inn, btop, score, innscore)

    def get_batter(self, batorder, btop):
        if(btop == 1):
            return [d for d in self.TeamLineup["AwayBatters"] if batorder == d["batOrder"]]
        else:
            return [d for d in self.TeamLineup["HomeBatters"] if batorder == d["batOrder"]]


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
            annotation = self.BntFoul(ball_data)

        elif ("구 타격" in text):
            annotation = self.Hit(ball_data)

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
        ball_loc = self.calc_ball_location(ball_data)

        ballCount = str(ball_data["ballcount"])
        Speed = str(ball_data["speed"])
        stuff = str(ball_data["stuff"])

        if(int(ballCount) >= 6):
            return cut(ballCount, Speed, stuff)
        else:
            return normalFoul(ballCount, Speed, stuff)


    def BntFoul(self, ball_data):
        ball_loc = self.calc_ball_location(ball_data)

        ballCount = str(ball_data["ballcount"])
        Speed = str(ball_data["speed"])
        stuff = str(ball_data["stuff"])

        return buntFoul(ballCount, Speed, stuff)

    def Hit(self, ball_data):
        ballCount = str(ball_data["ballcount"])
        Speed = str(ball_data["speed"])
        stuff = str(ball_data["stuff"])

        return hit(ballCount, Speed, stuff)


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

'''
    삼진 아웃, 볼넷, 고의4구, 몸에 맞는 볼
    
    
    중견수 플라이 아웃
    3루수 땅볼 아웃 (3루수 -> 1루수 송구아웃)
    3루수 병살타 아웃 (3루수->2루수->1루수 송구아웃)
    포스아웃 (3루수->2루수 2루 터치아웃)
    투수 희생번트 아웃
    2루수 라인드라이브 아웃
    포수 스트라이크 낫 아웃
    
    좌익수 앞 1루타
    우중간 1루타
    우익수 뒤 2루타
    유격수 앞 땅볼로 출루
    
    2루수 실책으로 출루 < 안함
    
'''
class Result():
    def __init__(self, GameInfo, TeamLineup):
        self.GameInfo = GameInfo
        self.TeamLineup = TeamLineup

    def set(self, relayText):
        text = relayText["liveText"]

        annotation = ""
        if("아웃" in text):
            annotation = self.out(relayText)

        elif("고의" in text or "볼넷" in text or "몸에" in text or "출루" in text or "루타" in text or "내야안타" in text or "홈런" in text): #출루, 안타, 홈런
            annotation = self.hit(relayText)

        elif("진루" in text or "홈인" in text):
            annotation = self.run(relayText)

        print(annotation)

    def run(self, relayText):
        text = relayText["liveText"]

        origin = "?"
        destination = "?"
        name = "?"

        text_ = text.split(" ")

        name = text_[1]
        origin = [i for i in text_ if ("주자" in i)][0][:2]

        if("홈인" in text):
            aScore = str(relayText["awayScore"])
            hScore = str(relayText["homeScore"])
            return runHome(origin, name, aScore, hScore)

        elif("도루" in text):
            destination = [i for i in text_ if ("까지" in i)][0][:2]
            return thiefBase(origin, destination, name)

        elif("진루" in text):
            destination = [i for i in text_ if ("까지" in i)][0][:2]
            if("실책" in text):
                return runError(origin, destination, name)
            else:
                return runBase(origin, destination, name)

    def hit(self, relayText):
        text = relayText["liveText"]

        batorder = relayText["batorder"]
        btop = relayText["btop"]

        batter = self.get_batter(batorder, btop)[0]
        name = str(batter["name"])

        if("볼넷" in text):
            return fourBall(name)

        elif("고의" in text):
            return intentionalBaseOnBalls(name)

        elif("몸에 맞는" in text):
            return hitByPitch(name)

        pos = "?"
        text_ = text.split(" ")
        if(text_[1] == ":"):
            pos = text.split(" ")[2]

        if("출루" in text):
            if("실책" in text):
                return errorWalk(name, pos)
            else:
                return groundballWalk(name, pos)

        elif ("내야안타" in text):
            return inFieldHit(name, pos)

        elif("1루타" in text):
            return outFieldHit(name, pos)

        elif ("2루타" in text):
            return outFieldDoubleHit(name, pos)

        elif ("3루타" in text):
            return outFieldTripleHit(name, pos)

        elif("홈런" in text):
            return HomeRun(name, pos)



    def out(self, relayText):
        """
            * 중견수 플라이 아웃
            * 3루수 땅볼 아웃 (3루수 -> 1루수 송구아웃)
            * 3루수 병살타 아웃 (3루수->2루수->1루수 송구아웃)
            * 포스아웃 (3루수->2루수 2루 터치아웃)
            * 투수 희생번트 아웃
            * 2루수 라인드라이브 아웃
            * 포수 스트라이크 낫 아웃
            * 도루실패아웃
        """

        text = relayText["liveText"]

        batorder = relayText["batorder"]
        btop = relayText["btop"]

        batter = self.get_batter(batorder, btop)[0]

        if ("삼진" in text):
            return strikeOut(batter)

        elif ("플라이" in text):
            return 1

        elif ("땅볼" in text):
            return 1

        elif ("라인드라이브" in text):
            return 1


    def get_batter(self, batorder, btop):
        if (btop == 1):
            return [d for d in self.TeamLineup["AwayBatters"] if batorder == d["batOrder"]]
        else:
            return [d for d in self.TeamLineup["HomeBatters"] if batorder == d["batOrder"]]


