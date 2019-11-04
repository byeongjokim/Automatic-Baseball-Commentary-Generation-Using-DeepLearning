import json
import random
from Web.annotation import *
from Web.set_onto import *


class Change():
    def __init__(self, GameInfo, TeamLineup, onto, resources):
        self.GameInfo = GameInfo
        self.TeamLineup = TeamLineup
        self.onto = onto
        self.resources = resources

    def set(self, relayText):
        text = relayText["liveText"]

        annotation = ""
        if ("번타자" in text):
            annotation = self.batterbox(relayText)
            self.resources.clear_strike_ball()

        elif ("교체" in text):
            annotation = self.change_player(relayText)

        elif ("변경" in text):
            annotation = self.change_position(relayText)

        elif ("종료" in text):
            annotation = self.fin_attack(relayText)
            self.resources.clear_strike_ball_out()

        self.resources.set_action("etc")
        return annotation

    def batterbox(self, relayText):
        batorder = relayText["batorder"]
        btop = relayText["btop"]

        batter = self.get_batter(batorder, btop)[0]
        self.resources.set_batter(batter["name"] + batter["pCode"])

        return batter_Start(batter)

    #########################이쪽 수정
    def change_player(self, relayText):
        btop = relayText["btop"]
        text = relayText["liveText"]
        text_ = text.split(" ")

        player_in = text_[4]
        player_out = text_[1]

        if ("투수" in text):
            player_in = self.get_pitcher_with_name(player_in, btop)[0]
            player_out = self.get_pitcher_with_name(player_out, btop)[0]
        else:
            if (self.get_batter_with_name(player_in, 0) != [] and self.get_batter_with_name(player_out, 0) != []):
                player_in = self.get_batter_with_name(player_in, 0)[0]
                player_out = self.get_batter_with_name(player_out, 0)[0]
            else:
                player_in = self.get_batter_with_name(player_in, 1)[0]
                player_out = self.get_batter_with_name(player_out, 1)[0]
                self.resources.change_LineUp(player_in, player_out)

        create_change(self.onto, self.GameInfo, player_in, player_out, self.resources.get_seq())
        self.resources.add_seq()

        return relayText["liveText"]

    def change_position(self, relayText):
        return relayText["liveText"]

    def fin_attack(self, relayText):
        inn = relayText["inn"]
        btop = relayText["btop"]
        score = {"homeScore": relayText["homeScore"], "awayScore": relayText["awayScore"]}
        innscore = {"homeInningScore": relayText["homeInningScore"], "awayInningScore": relayText["awayInningScore"]}

        self.resources.set_inn(create_inn(self.onto, self.GameInfo, inn, btop), btop)
        return inn_end(self.GameInfo, inn, btop, score, innscore)

    def get_batter(self, batorder, btop):
        if (btop == 1):
            return [d for d in self.TeamLineup["AwayBatters"] if batorder == d["batOrder"]]
        else:
            return [d for d in self.TeamLineup["HomeBatters"] if batorder == d["batOrder"]]

    def get_batter_with_name(self, name, btop):
        if (btop == 1):
            return [d for d in self.TeamLineup["AwayBatters"] if name == d["name"]]
        else:
            return [d for d in self.TeamLineup["HomeBatters"] if name == d["name"]]

    def get_pitcher_with_name(self, name, btop):
        if (btop == 0):
            return [d for d in self.TeamLineup["AwayPitchers"] if name == d["name"]]
        else:
            return [d for d in self.TeamLineup["HomePitchers"] if name == d["name"]]


class PitchingBatting():
    def __init__(self, GameInfo, TeamLineup, onto, resources):
        self.GameInfo = GameInfo
        self.TeamLineup = TeamLineup
        self.onto = onto
        self.resources = resources

        self.num_BatterBox = 1

    def set(self, relayText, ball_data):
        text = relayText["liveText"]

        inn = relayText["inn"]
        btop = relayText["btop"]
        pitchId = relayText["pitchId"]

        ballcount = relayText["ballcount"]

        batterName = ball_data["batterName"]
        batorder = relayText["batorder"]
        ilsun = relayText["ilsun"]
        batter = self.get_batter_with_name(batterName, btop)[0]

        pitcherName = ball_data["pitcherName"]
        pitcher = self.get_pitcher_with_name(pitcherName, btop)[0]
        stuff = ball_data["stuff"]

        inn_instance = self.resources.get_inn()
        stay = [self.get_batter_with_order(relayText["base1"], btop),
                self.get_batter_with_order(relayText["base2"], btop),
                self.get_batter_with_order(relayText["base3"], btop)]

        if (ballcount == 1):  # 1구 -> create batterbox instance

            batter_box, p, b = create_batterbox(self.onto, self.GameInfo, self.num_BatterBox, batter, pitcher, batorder,
                                                stay, inn_instance, btop)
            self.resources.set_batterbox(batter_box, p, b)
            self.resources.clear_strike_ball_out()
            self.num_BatterBox = self.num_BatterBox + 1

        annotation = ""
        if ("구 스트라이크" in text):
            annotation = self.Strike(ball_data)
            create_pitchingbatting(self.onto, self.GameInfo, "strike", self.resources.get_batterbox(),
                                   self.resources.get_seq())
            self.resources.add_seq()
            self.resources.set_strike_ball_out(strike=True)
            self.resources.set_action("strike")

        elif ("구 헛스윙" in text):
            annotation = self.Swing(ball_data)
            create_pitchingbatting(self.onto, self.GameInfo, "strike", self.resources.get_batterbox(),
                                   self.resources.get_seq())
            self.resources.add_seq()
            self.resources.set_strike_ball_out(strike=True)
            self.resources.set_action("strike")

        elif ("구 볼" in text):
            annotation = self.Ball(ball_data)
            create_pitchingbatting(self.onto, self.GameInfo, "ball", self.resources.get_batterbox(),
                                   self.resources.get_seq())
            self.resources.set_strike_ball_out(ball=True)
            self.resources.add_seq()
            self.resources.set_action("ball")

        elif ("구 파울" in text):
            annotation = self.Foul(ball_data)
            create_pitchingbatting(self.onto, self.GameInfo, "foul", self.resources.get_batterbox(),
                                   self.resources.get_seq())
            self.resources.add_seq()
            self.resources.set_strike_ball_out(foul=True)
            self.resources.set_action("foul")

        elif ("구 번트파울" in text):
            annotation = self.BntFoul(ball_data)
            create_pitchingbatting(self.onto, self.GameInfo, "foul", self.resources.get_batterbox(),
                                   self.resources.get_seq())
            self.resources.add_seq()
            self.resources.set_strike_ball_out(foul=True)
            self.resources.set_action("foul")

        elif ("구 타격" in text):
            annotation = self.Hit(ball_data)
            self.resources.set_action("hit")

        return annotation

    def Strike(self, ball_data):
        ball_loc = self.calc_ball_location(ball_data)

        ballCount = str(ball_data["ballcount"])
        Speed = str(ball_data["speed"])
        stuff = str(ball_data["stuff"])

        if ("ID" == ball_loc or "IU" == ball_loc or "OD" == ball_loc or "OU" == ball_loc):
            return difficultCourseStrike(ballCount, Speed, stuff)

        elif ("MM" == ball_loc):
            return easyCourseStrike(ballCount, Speed, stuff)

        elif ("IM" == ball_loc):
            return insideStrike(ballCount, Speed, stuff)

        elif ("OM" == ball_loc):
            return outsideStrike(ballCount, Speed, stuff)

        elif ("MD" == ball_loc):
            return downsideStrike(ballCount, Speed, stuff)

        elif ("MU" == ball_loc):
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

        if (int(ballCount) >= 6):
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

    def get_batter_with_name(self, name, btop):
        if (btop == 1):
            return [d for d in self.TeamLineup["AwayBatters"] if name == d["name"]]
        else:
            return [d for d in self.TeamLineup["HomeBatters"] if name == d["name"]]

    def get_pitcher_with_name(self, name, btop):
        if (btop == 1):
            return [d for d in self.TeamLineup["HomePitchers"] if name == d["name"]]
        else:
            return [d for d in self.TeamLineup["AwayPitchers"] if name == d["name"]]

    def get_batter_with_order(self, no, btop):
        if (btop == 1):
            return [d for d in self.TeamLineup["AwayBatters"] if no == d["batOrder"]]
        else:
            return [d for d in self.TeamLineup["HomeBatters"] if no == d["batOrder"]]


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
    def __init__(self, GameInfo, TeamLineup, onto, resources):
        self.GameInfo = GameInfo
        self.TeamLineup = TeamLineup
        self.onto = onto
        self.resources = resources

    def set(self, relayText):
        text = relayText["liveText"]

        annotation = ""

        if ("비디오" in text):
            annotation = text
            self.resources.set_action("etc")

        elif ("아웃" in text):
            annotation = self.out(relayText)
            self.resources.set_strike_ball_out(out=True)
            self.resources.set_action("out")

        elif ("고의" in text or "볼넷" in text or "몸에" in text or "출루" in text or "루타" in text or "내야안타" in text or "홈런" in text):  # 출루, 안타, 홈런
            annotation = self.hit(relayText)
            if "홈런" in text:
                self.resources.set_action("homerun")
            else:
                self.resources.set_action("hits")

        elif ("진루" in text or "홈인" in text):
            annotation = self.run(relayText)
            self.resources.set_action("etc")

        try:
            text = text[:text.index("(")]
            return text

        except:
            return text

    def run(self, relayText):
        text = relayText["liveText"]

        btop = relayText["btop"]

        text_ = text.split(" ")

        name = text_[1]
        runner = self.get_batter_with_name(name, btop)[0]
        origin = [i for i in text_ if ("주자" in i)][0][:2]

        if ("홈인" in text):
            aScore = str(relayText["awayScore"])
            hScore = str(relayText["homeScore"])

            create_run(self.onto, self.GameInfo, "homein", "home", runner, self.resources.get_batterbox(),
                       self.resources.get_seq())
            return runHome(origin, name, aScore, hScore)

        elif ("도루" in text):
            destination = [i for i in text_ if ("까지" in i)][0][:2]

            create_run(self.onto, self.GameInfo, "steal", destination[0], runner, self.resources.get_batterbox(),
                       self.resources.get_seq())
            self.resources.add_seq()

            return thiefBase(origin, destination, name)

        elif ("진루" in text):
            destination = [i for i in text_ if ("까지" in i)][0][:2]

            create_run(self.onto, self.GameInfo, "runBase", destination[0], runner, self.resources.get_batterbox(),
                       self.resources.get_seq())

            if ("실책" in text):
                return runError(origin, destination, name)
            else:
                return runBase(origin, destination, name)

    def hit(self, relayText):
        text = relayText["liveText"]

        batorder = relayText["batorder"]
        btop = relayText["btop"]

        batter = self.get_batter(batorder, btop)[0]
        name = str(batter["name"])

        if ("볼넷" in text):
            create_hit(self.onto, self.GameInfo, "fourball", self.resources.get_batterbox(), self.resources.get_seq())
            self.resources.add_seq()
            return fourBall(name)

        elif ("고의" in text):
            create_hit(self.onto, self.GameInfo, "fourball", self.resources.get_batterbox(), self.resources.get_seq())
            self.resources.add_seq()
            return intentionalBaseOnBalls(name)

        elif ("몸에 맞는" in text):
            create_hit(self.onto, self.GameInfo, "hitbypitch", self.resources.get_batterbox(), self.resources.get_seq())
            self.resources.add_seq()
            return hitByPitch(name)

        pos = "?"
        text_ = text.split(" ")
        if (text_[1] == ":"):
            pos = text.split(" ")[2]

        if ("출루" in text):
            if ("실책" in text):
                create_hit(self.onto, self.GameInfo, "errorwalk", self.resources.get_batterbox(),
                           self.resources.get_seq())
                self.resources.add_seq()
                return errorWalk(name, pos)
            else:
                create_hit(self.onto, self.GameInfo, "groundballwalk", self.resources.get_batterbox(),
                           self.resources.get_seq())
                self.resources.add_seq()
                return groundballWalk(name, pos)

        elif ("내야안타" in text):
            create_hit(self.onto, self.GameInfo, "singlehit", self.resources.get_batterbox(), self.resources.get_seq())
            self.resources.add_seq()
            return inFieldHit(name, pos)

        elif ("1루타" in text):
            create_hit(self.onto, self.GameInfo, "singlehit", self.resources.get_batterbox(), self.resources.get_seq())
            self.resources.add_seq()
            return outFieldHit(name, pos)

        elif ("2루타" in text):
            create_hit(self.onto, self.GameInfo, "doublehit", self.resources.get_batterbox(), self.resources.get_seq())
            self.resources.add_seq()
            return outFieldDoubleHit(name, pos)

        elif ("3루타" in text):
            create_hit(self.onto, self.GameInfo, "triplehit", self.resources.get_batterbox(), self.resources.get_seq())
            self.resources.add_seq()
            return outFieldTripleHit(name, pos)

        elif ("홈런" in text):
            create_hit(self.onto, self.GameInfo, "homerun", self.resources.get_batterbox(), self.resources.get_seq())
            self.resources.add_seq()
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
            create_out(self.onto, self.GameInfo, "strikeout", self.resources.get_batterbox(), self.resources.get_seq())
            self.resources.add_seq()
            return strikeOut(batter)

        elif ("플라이" in text):
            create_out(self.onto, self.GameInfo, "fly", self.resources.get_batterbox(), self.resources.get_seq())
            self.resources.add_seq()

            pos = "?"
            text_ = text.split(" ")
            if (text_[1] == ":"):
                pos = text.split(" ")[2]

            return flyOut(batter, pos)

        elif ("땅볼" in text):
            create_out(self.onto, self.GameInfo, "outinbase", self.resources.get_batterbox(), self.resources.get_seq())
            self.resources.add_seq()

            pos = "?"
            text_ = text.split(" ")
            if (text_[1] == ":"):
                pos = text.split(" ")[2]

            return groundBallOut(batter, pos)

        elif ("라인드라이브" in text):
            create_out(self.onto, self.GameInfo, "fly", self.resources.get_batterbox(), self.resources.get_seq())
            self.resources.add_seq()

            pos = "?"
            text_ = text.split(" ")
            if (text_[1] == ":"):
                pos = text.split(" ")[2]

            return lineDriveOut(batter, pos)

        elif ("병살" in text):
            create_out(self.onto, self.GameInfo, "double play", self.resources.get_batterbox(),
                       self.resources.get_seq())
            self.resources.add_seq()

            pos = "?"
            text_ = text.split(" ")
            if (text_[1] == ":"):
                pos = text.split(" ")[2]

            return doublePlayedOut(batter, pos)

        elif ("낫 아웃" in text):
            create_out(self.onto, self.GameInfo, "strikeout", self.resources.get_batterbox(), self.resources.get_seq())
            self.resources.add_seq()

            pos = "?"
            text_ = text.split(" ")
            if (text_[1] == ":"):
                pos = text.split(" ")[2]

            return strikeNotOut(batter, pos)

        elif ("희생번트" in text):
            create_out(self.onto, self.GameInfo, "outinbase", self.resources.get_batterbox(), self.resources.get_seq())
            self.resources.add_seq()

            pos = "?"
            text_ = text.split(" ")
            if (text_[1] == ":"):
                pos = text.split(" ")[2]

            return sacrificeBunt(batter, pos)

        else:
            name = text.split(" ")[1]
            create_out(self.onto, self.GameInfo, "tagNforceOut", self.resources.get_batterbox(),
                       self.resources.get_seq())
            self.resources.add_seq()

            return tagNforceOut(name)

    def get_batter(self, batorder, btop):
        if (btop == 1):
            return [d for d in self.TeamLineup["AwayBatters"] if batorder == d["batOrder"]]
        else:
            return [d for d in self.TeamLineup["HomeBatters"] if batorder == d["batOrder"]]

    def get_batter_with_name(self, name, btop):
        if (btop == 1):
            return [d for d in self.TeamLineup["AwayBatters"] if name == d["name"]]
        else:
            return [d for d in self.TeamLineup["HomeBatters"] if name == d["name"]]
