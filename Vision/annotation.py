import rdflib
import settings

class Annotation():
    def __init__(self):
        self.g = rdflib.Graph()
        self.g.load(settings.OWL_FILE)
        self.uri = settings.OWL_URI
        self._set_object_properties()
        self._set_data_properties()

        self.home_atmosphere = 0

    def _set_instance(self):

        return 1

    def _set_object_properties(self):
        self.inGame = rdflib.URIRef(self.uri + "inGame")

        self.fromPitcher = rdflib.URIRef(self.uri + "fromPitcher")
        self.toHitter = rdflib.URIRef(self.uri + "toHitter")

        self.stayIn1stBase = rdflib.URIRef(self.uri + "stayIn1stBase")
        self.stayIn2ndBase = rdflib.URIRef(self.uri + "stayIn2ndBase")
        self.stayIn3rdBase = rdflib.URIRef(self.uri + "stayIn3rdBase")

        self.result = rdflib.URIRef(self.uri + "result")

    def _set_data_properties(self):
        self.thisERA = rdflib.URIRef(self.uri + "thisERA")
        self.thisAVG = rdflib.URIRef(self.uri + "thisAVG")

    def reload(self):
        try:
            self.g.load(settings.OWL_FILE)
        except:
            pass

    def get_atmosphere(self, strike_ball_out, b=None, p=None):
        strike, ball, out = strike_ball_out

        annotation_atmosphere = []
        if (strike == 2):
            if(p):
                p = self.get_player_name(p)
                annotation_atmosphere.append("투 스트라이크 상황 " + p + " 투수 스트라이크 아웃 잡아 낼 수 있을까요?")
                annotation_atmosphere.append(p + " 투수 삼진아웃 까지 스트라이크 하나 남겨두고 있습니다.")
            if(b):
                b = self.get_player_name(b)
                annotation_atmosphere.append(b + " 타자 투 스트라이크 위기입니다.")
                annotation_atmosphere.append(b + " 타자 핀치에 몰렸습니다.")
                annotation_atmosphere.append(b + " 타자 위기입니다.")

        elif(ball == 3):
            if(p):
                p = self.get_player_name(p)
                annotation_atmosphere.append("쓰리볼인 상황 " + p + " 투수 좋은 공을 던져야 합니다.")
            if(b):
                b = self.get_player_name(b)
                annotation_atmosphere.append(b + " 타자 포볼로 출루 할 수 있는 기회입니다.")
                annotation_atmosphere.append(b + " 타자 출루까지 볼 하나 남겨두고 있습니다.")
                annotation_atmosphere.append("쓰리볼인 상황 " + b + " 타자 공을 잘 걸러낼 수 있을까요?")

        elif (out == 2):
            if(p):
                p = self.get_player_name(p)
                annotation_atmosphere.append(p + " 투수 쓰리아웃까지 아웃 하나 남았습니다.")
            if(b):
                b = self.get_player_name(b)
                annotation_atmosphere.append(b + " 타자 아웃 당하면 이번 공격 마무리 됩니다.")

        return annotation_atmosphere

    def search_team(self, gameinfo, btop):
        home_Fname = gameinfo[3][0]
        away_Fname = gameinfo[3][1]

        annotation = []
        if(btop == 0): #away defense
            new = [home_Fname + "팀 공격을 하고 있습니다.",
                   "홈팀 공격하고 있습니다."
                   ]

        else:

            new = [away_Fname + "팀 공격을 하고 있습니다.",
                   "어웨이팀 공격하고 있습니다."
                   ]

        annotation = annotation + new
        return annotation

    def search_gameInfo(self, gameCode, inn, score, gameinfo):
        homescore = score[0]
        awayscore = score[1]

        if(homescore < awayscore):
            about_atmosphere = "어웨이팀이 유리한 가운데"
        elif(homescore > awayscore):
            about_atmosphere = "홈팀이 유리한 가운데 "
        else:
            about_atmosphere = "비등비등한 분위기 속에"

        stadium = gameinfo[0]
        date = gameinfo[1]
        home_full_name = gameinfo[3][0]
        away_full_name = gameinfo[3][1]

        ####here
        annotation = [
            str(stadium) + "에서 경기 진행중입니다.",
            #"현재 경기 스코어 " + str(homescore) + " 대 " + str(awayscore) + " 입니다.",
            str(inn).split("_")[-1] + " 경기 스코어 " + str(homescore) + " 대 " + str(awayscore) + " 진행중입니다.",
            #home_full_name + " 대 " + away_full_name + " 현재 " + str(homescore) + " 대 " + str(awayscore) + " 진행중입니다.",
        ]

        annotation = annotation + [about_atmosphere + " " + i for i in annotation]

        return annotation

    def search_pitcher(self, gameCode, p, strike_ball_out):
        pitcher = rdflib.URIRef(self.uri + p)
        thisGame = rdflib.URIRef(self.uri + gameCode)

        query = "SELECT ?o WHERE {?pitcher ?thisERA ?o}"
        r = self.g.query(query, initBindings={"pitcher": pitcher, "thisERA":self.thisERA})

        era = 0
        for row in r:
            era = row[0]

        query = "SELECT ?o WHERE {?s ?fromPitcher ?pitcher . ?s ?inGame ?thisGame . ?s ?result ?o}"
        r = self.g.query(query, initBindings={"fromPitcher": self.fromPitcher, "pitcher": pitcher,
                                         "inGame": self.inGame, "thisGame": thisGame,
                                         "result": self.result})
        strikeout = 0
        baseonballs = 0

        #print("len r : ", str(len(r)))
        for row in r:
            if ("Strikeout" in row[0]):
                strikeout = strikeout + 1
            if ("BaseOnBalls" in row[0]):
                baseonballs = baseonballs + 1

        annotation_atmosphere = self.get_atmosphere(strike_ball_out, p=p)

        p = self.get_player_name(p)
        annotation = [
            p + " 투수 오늘 경기 " + str(len(r) + 1) + "번째 타석에서 공을 던지고 있습니다.",
            p + " 투수 오늘 경기 " + str(strikeout) + "개의 삼진을 잡아내고 있습니다.",
            p + " 투수 오늘 경기 " + str(baseonballs) + "개의 포볼로 타자를 진루 시켰습니다.",
            p + " 투수 이번 시즌 " + str(era) + "의 평균 자책점을 기록하고 있습니다.",
            p + " 투수 과연 어떤 공을 던질까요?",
            "투수 오늘 경기 " + str(len(r) + 1) + "번째 타석에서 공을 던지고 있습니다.",
            "투수 오늘 경기 " + str(strikeout) + "개의 삼진을 잡아내고 있습니다.",
            "투수 오늘 경기 " + str(baseonballs) + "개의 포볼로 타자를 진루 시켰습니다.",
            "투수 이번 시즌 " + str(era) + "의 평균 자책점을 기록하고 있습니다.",
            "투수 과연 어떤 공을 던질까요?",
        ]

        return annotation + annotation_atmosphere

    def search_batter(self, gameCode, b, strike_ball_out):
        thisGame = rdflib.URIRef(self.uri + gameCode)
        batter = rdflib.URIRef(self.uri + b)

        query = "SELECT ?o where {?batter ?thisAVG ?o}"
        r = self.g.query(query, initBindings={"batter": batter, "thisAVG": self.thisAVG})

        avg = 0
        for row in r:
            avg = row[0]

        query = "SELECT ?o where {?s ?toHitter ?hitter . ?s ?inGame ?thisGame . ?s ?result ?o } order by ?s"
        r = self.g.query(query, initBindings={"toHitter": self.toHitter, "hitter": batter, "inGame": self.inGame, "thisGame": thisGame,
                                         "result": self.result})
        this_game_count = len(r)
        batter_history = []
        for row in r:
            batter_history.append(
                self.change_result_history_to_korean(row[0].split("#")[1].split("_")[1])
            )

        query = "SELECT ?o where {?s ?toHitter ?hitter . ?s ?result ?o . ?s ?stayIn1stBase ?o1} order by ?s"
        r = self.g.query(query, initBindings={"toHitter": self.toHitter, "hitter": batter,
                                         "inGame": self.inGame, "thisGame": thisGame,
                                         "result": self.result, "stayIn1stBase": self.stayIn1stBase})

        batter_history_when1st = []
        for row in r:
            batter_history_when1st.append(
                self.change_result_history_to_korean(row[0].split("#")[1].split("_")[1])
            )

        annotation_atmosphere = self.get_atmosphere(strike_ball_out, b=b)

        b = self.get_player_name(b)
        annotation = [
            b + " 타자",
            b + " 타자의 오늘 " + str(this_game_count + 1) + "번째 타석입니다.",
            b + " 타자는 이번 시즌 " + str(avg) + "의 평균 타율을 기록하고 있습니다.",
            b + " 타자 이번 타석 안타를 기록 할 수 있을까요?",
            ###here
            #"타자의 오늘 " + str(this_game_count + 1) + "번째 타석입니다.",
            #"타자는 이번 시즌 " + str(avg) + "의 평균 타율을 기록하고 있습니다.",
            #"타자 이번 타석 안타를 기록 할 수 있을까요?",
        ]
        if (batter_history):
            annotation.append(b + " 타자 오늘 " + str(this_game_count + 1) + "번째 타석입니다, " + ", ".join(_ for _ in batter_history) + "을 기록하고 있습니다.")
            annotation.append(b + " 타자 저번 타석, " + str(batter_history[-1]) + "을 기록하였습니다.")

        if (batter_history_when1st):
            annotation.append(b + " 타자 1루 주자가 있는 상황에서 " + ", ".join(_ for _ in batter_history_when1st) + "을 기록하고 있습니다.")
            annotation.append("오늘 1루 주자가 있는 타석에서 " + b + " 타자 최근, " + str(batter_history_when1st[-1]) + "을 기록하였습니다.")

        return annotation + annotation_atmosphere

    def search_pitcherbatter(self, gameCode, p, b, strike_ball_out):
        thisGame = rdflib.URIRef(self.uri + gameCode)
        batter = rdflib.URIRef(self.uri + b)
        pitcher = rdflib.URIRef(self.uri + p)

        avg = 0
        era = 0
        query = "SELECT ?o where {?s ?p ?o}"
        r = self.g.query(query, initBindings={"s": batter, "p": self.thisAVG})
        for row in r:
            avg = row[0]

        r = self.g.query(query, initBindings={"s": pitcher, "p": self.thisERA})
        for row in r:
            era = row[0]

        query = "SELECT ?o where {?s ?inGame ?thisGame . ?s ?toHitter ?hitter . ?s ?fromPitcher ?pitcher . ?s ?result ?o} order by desc(?s)"
        r = self.g.query(query, initBindings={
            "inGame": self.inGame, "thisGame": thisGame,
            "toHitter": self.toHitter, "hitter": batter,
            "fromPitcher": self.fromPitcher, "pitcher": pitcher,
            "result": self.result
        })

        result_history = []
        strikeout = 0
        getonbase = 0
        for row in r:
            result_history.append(
                self.change_result_history_to_korean(row[0].split("#")[1].split("_")[1])
            )

            if ("Strikeout" in row[0]):
                strikeout = strikeout + 1
            if ("BaseOnBalls" in row[0]):
                getonbase = getonbase + 1
            if ("HitByPitch" in row[0]):
                getonbase = getonbase + 1
            if ("Double" in row[0]):
                getonbase = getonbase + 1
            if ("HomeRun" in row[0]):
                getonbase = getonbase + 1
            if ("Triple" in row[0]):
                getonbase = getonbase + 1
            if ("SingleHit" in row[0]):
                getonbase = getonbase + 1

        b = self.get_player_name(b)
        p = self.get_player_name(p)
        annotation = [
            b + " 타자 " + p + " 투수의 신경전 속에 " + b + " 타자는 이번시즌 " + str(avg) + "의 펑균 타율을 기록하고 있습니다.",
            b + " 타자 " + p + " 투수의 신경전 속에 " + p + " 투수는 이번시즌 " + str(era) + "의 평균 자책점을 기록하고 있습니다.",
            b + " 타자 " + p + " 투수를 상대로 오늘 " + str(getonbase) + "개의 안타 기록 하였습니다.",
            p + " 투수 " + b + " 타자를 상대로 오늘 경기 " + str(getonbase) + "개의 안타를 허용 하였습니다.",
            p + " 투수 " + b + " 타자를 상대로 오늘 경기 " + str(strikeout) + "개의 스트라이크 아웃을 잡아냈습니다.",
            "투수와 타자 사이에 팽팽한 긴장감이 감지됩니다.",
        ]
        if (result_history):
            annotation = annotation + [b + " 타자 " + p + " 투수를 상대로 오늘 " + ", ".join(_ for _ in result_history) + "을 기록 하였습니다."]

        return annotation

    def search_runner(self, batterbox):
        annotation = []
        first_runner_annotation, first_runner = self.search_first_runner(batterbox)
        second_runner_annotation, second_runner = self.search_second_runner(batterbox)
        third_runner_annotation, third_runner = self.search_third_runner(batterbox)

        annotation = annotation + first_runner_annotation
        annotation = annotation + second_runner_annotation
        annotation = annotation + third_runner_annotation

        score_zone = ""
        result = ""
        if(first_runner):
            result = result + "1루에는 " + str(first_runner) + "선수 "
        if (second_runner):
            result = result + "2루에는 " + str(second_runner) + "선수 "
            score_zone = score_zone + str(second_runner) + "선수 "

        if (third_runner):
            result = result + "3루에는 " + str(third_runner) + "선수 "
            score_zone = score_zone + str(third_runner) + "선수 "

        if not(result == ""):
            result = result + "(이)가 나가있습니다."
            annotation.append(result)

        if not(score_zone == ""):
            score_zone = score_zone + "득점권에 나가있습니다."
            annotation.append(score_zone)

        return annotation


    def search_first_runner(self, batterbox):
        bbox = str(batterbox).split(".")[-1]
        batterbox = rdflib.URIRef(self.uri + bbox)

        query = "select ?o where {?batterbox ?stayIn1stBase ?o}"
        r = self.g.query(query, initBindings={
            "batterbox": batterbox,
            "?stayIn1stBase": self.stayIn1stBase
        })

        annotation = []
        runner = None

        for row in r:
            runner = row[0].split("#")[-1]

        if(runner):
            runner = self.get_player_name(str(runner))
            annotation = [
                "1루에는 " + str(runner) + "선수가 주자로 나가있습니다.",
                "1루에는 " + str(runner) + "선수가 나가있습니다.",
            ]

        return annotation, runner

    def search_second_runner(self, batterbox):
        bbox = str(batterbox).split(".")[-1]
        batterbox = rdflib.URIRef(self.uri + bbox)

        query = "select ?o where {?batterbox ?stayIn2ndBase ?o}"
        r = self.g.query(query, initBindings={
            "batterbox": batterbox,
            "?stayIn2ndBase": self.stayIn2ndBase
        })

        annotation = []
        runner = None

        for row in r:
            runner = row[0].split("#")[-1]

        if (runner):
            runner = self.get_player_name(str(runner))
            annotation = [
                "2루에 " + str(runner) + "선수가 주자로 나가있습니다.",
                "2루에 " + str(runner) + "선수가 나가있습니다.",
            ]

        return annotation, runner

    def search_third_runner(self, batterbox):
        bbox = str(batterbox).split(".")[-1]
        batterbox = rdflib.URIRef(self.uri + bbox)

        query = "select ?o where {?batterbox ?stayIn3rdBase ?o}"
        r = self.g.query(query, initBindings={
            "batterbox": batterbox,
            "?stayIn3rdBase": self.stayIn3rdBase
        })

        annotation = []
        runner = None

        for row in r:
            runner = row[0].split("#")[-1]

        if (runner):
            runner = self.get_player_name(str(runner))
            annotation = [
                "3루에 " + str(runner) + "선수가 주자로 나가있습니다.",
                "3루에 " + str(runner) + "선수가 나가있습니다.",
            ]

        return annotation, runner

    def get_player_name(self, name):
        return "".join([s for s in list(name) if not s.isdigit()])

    def change_result_history_to_korean(self, h):
        result = ["BaseByError", "BaseOnBalls", "HitByPitch", "Double", "HomeRun", "Triple", "SingleHit", "Fly", "OutInBase", "Strikeout", "Out", "GetOnBase"]
        korean = ["실책 출루", "포 볼", "데드 볼", "2루타", "홈런", "3루타", "1루타", "플라이 아웃", "땅볼 아웃", "스트라이크 아웃", "아웃", "출루"]

        return korean[result.index(h)]

    def get_motion_annotation(self, scene_label, motion_label, who=None, resource=None):
        anno = []
        if (who):
            if(who == "pitcher" and motion_label == 3):
                pitcher = self._get_player_name(resource.get_pitcher())
                anno = [
                    str(pitcher) + " 투수 공을 던졌습니다.",
                    "공을 던졌습니다.",
                    str(pitcher) + " 투수 타자를 향해 힘껏 공을 던졌습니다.",
                ]

            if(who == "batter" and motion_label == 0):
                batter = self.get_player_name(resource.get_batter())
                anno = [
                    str(batter) + "타자 배트를 휘둘렀습니다.",
                    str(batter) + "타자 힘차게 배트를 휘둘렀습니다.",
                    "배트를 휘둘렀습니다.",
                    "스윙!",
                ]
        else:
            position = ""
            position_ = ""
            if (scene_label == 5):  # first base
                position = "1st"
                position_ = "1루수"
            if (scene_label == 8):  # second base
                position = "2nd"
                position_ = "2루수"
            if (scene_label == 10):  # third base
                position = "3rd"
                position_ = "3루수"
            if (scene_label == 6):  # center outfield
                position = "COF"
                position_ = "중견수"
            if (scene_label == 7):  # right outfield
                position = "ROF"
                position_ = "우익수"
            if (scene_label == 11):  # left outfield
                position = "LOF"
                position_ = "좌익수"
            if (scene_label == 12):  # ss
                position = "ss"
                position_ = "유격수"

            if(position):
                player = resource.get_player_with_position(position)
                player = self._get_player_name(player)
                anno = []
                if (motion_label == 2):
                    anno = [position_ + " 송구 하였습니다.", player + "선수 송구 하였습니다.",
                            position_ + " " + player + "선수 송구 하였습니다."]
                elif (motion_label == 5):
                    anno = [position_ + " 공을 잡았습니다.", player + "선수 공을 잡았습니다.",
                            position_ + " " + player + "선수 공을 잡았습니다."]
                elif (motion_label == 6 or motion_label == 7):
                    anno = [position_ + " 쪽 입니다.",
                            player + "선수 쪽 입니다.",
                            position_ + " " + player + "선수 쪽 입니다.",
                            position_ + " " + player + "선수"]
        return anno

    def get_situation_annotation(self, situation_label, scene_label):
        anno = []
        if(((scene_label > 4 and scene_label < 13) and scene_label != 9) and (situation_label < 6 and situation_label > 2)):
            print("hit or ground or flying", situation_label)
            if(situation_label == 3):
                anno = anno + ["타자 친공, 안타가 될 수 있을까요?"]

            elif(situation_label == 4):
                anno = anno + ["타자 플라이아웃 인가요?",
                               "타자가 친공 높이 떴습니다."
                               ]
            elif(situation_label == 5):
                anno = anno + ["타자가 친공 낮게 굴러갑니다.",
                               "땅볼 처리 되나요?"]

        if((scene_label < 3 ) and (situation_label < 3)):
            if(situation_label == 0):
                anno = anno + ["스트라이크 인가요?",
                               "스트라이크로 예상됩니다."]
            elif(situation_label == 1):
                anno = anno + ["볼로 판정 될 것 같습니다."]

        return anno

    def _get_player_name(self, name):
        return "".join([s for s in list(name) if not s.isdigit()])