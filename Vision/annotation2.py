import rdflib
import settings
from itertools import product

class Annotation():
    def __init__(self):
        self.rdf = rdflib.Graph()
        self.rdf.load(settings.OWL_FILE)

        self.uri = settings.OWL_URI

        self._set_object_properties()
        self._set_data_properties()

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
            self.rdf.load(settings.OWL_FILE)
        except:
            pass

    def get_situation(self, gameinfo, inn, score, sbo):
        """
        0 회 0 스트라이크 0 볼 0 아웃 상황에서
        0회에서
        0 스트라이크 0 볼 상황에서
        0 아웃 상황에서

        어웨이팀이 유리한 가운데
        홈팀이 유리한 가운데
        비등한 분위기 속에
        0 대 0 상황에서
        """
        inn = str(inn).split("_")[-1]
        homescore = score[0]
        awayscore = score[1]

        stadium = gameinfo[0]
        date = gameinfo[1]
        home_full_name = gameinfo[3][0]
        away_full_name = gameinfo[3][1]

        situation = [
            #inn + " " + str(sbo[0]) + " 스트라이크 " + str(sbo[1]) + " 볼 " + str(sbo[2]) + " 아웃 상황, ",
            inn + " 진행중에, ",
            #str(sbo[0]) + " 스트라이크 " + str(sbo[1]) + " 볼 " + str(sbo[2]) + " 아웃 상황, ",
            str(sbo[0]) + " 스트라이크 " + str(sbo[1]) + " 볼 상황, ",
            #str(sbo[2]) + " 아웃 상황, ",

            str(stadium) + "에서 진행 중인 경기, ",
            str(stadium) + "에서 진행 중인 " + str(home_full_name) + " 대 " + str(away_full_name) + "경기, ",
            inn + " " + str(home_full_name) + " 대 " + str(away_full_name) + "경기, ",
            inn + " 경기 스코어 " + str(homescore) + " 대 " + str(awayscore) + " 진행 중, ",
            ""
        ]

        if(homescore < awayscore):
            situation.append("어웨이팀이 유리한 가운데, ")
        elif(homescore > awayscore):
            situation.append("홈팀이 유리한 가운데, ")
        else:
            situation.append("비등비등한 분위기 속에, ")

        return situation

    def search_runner(self, batterbox):
        bbox = str(batterbox).split(".")[-1]
        batterbox = rdflib.URIRef(self.uri + bbox)
        runner = [None, None, None]

        for ind, i in enumerate(["stayIn1stBase", "stayIn2ndBase", "stayIn3rdBase"]):
            query = "select ?o where {?batterbox ?%s ?o}" %i
            r = self.rdf.query(query, initBindings={
                "batterbox": batterbox,
                "?stayIn1stBase": self.stayIn1stBase,
                "?stayIn2ndBase": self.stayIn2ndBase,
                "?stayIn3rdBase": self.stayIn3rdBase,
            })
            if(r):
                runner[ind] = list(r)[0][0]

        return runner

    def about_batterbox(self, gameCode, gameinfo, inn, score, batterbox, sbo, pitcher, hitter, isaboutpitcher, isabouthitter, isaboutrunner):
        bbox = str(batterbox).split(".")[-1]
        batterbox_uri = rdflib.URIRef(self.uri + bbox)
        thisGame = rdflib.URIRef(self.uri + gameCode)
        pitcher_name = self.get_player_name(pitcher)
        pitcher = rdflib.URIRef(self.uri + pitcher)
        hitter_name = self.get_player_name(hitter)
        hitter = rdflib.URIRef(self.uri + hitter)

        situation = self.get_situation(gameinfo=gameinfo, inn=inn, score=score, sbo=sbo)
        annotation = []
        if(isaboutpitcher):
            """
                이름 추가해서
                    상황 추가해서
                        0 팀 소속 0 투수
                        투수 오늘 경기 0 번째 타석에서 공을 던집니다.
                        투수 오늘 경기 0 번째 타자를 상대하고 있습니다.
                        투수 오늘 경기 0 개의 삼진을 잡아내고 있습니다.
                        투수 오늘 경기 0 개의 포볼로 타자 출루 시켰습니다.
                        투수 오늘 경기 0 개의 플라이 아웃으로 타자 잡아냈습니다.
                        투수 오늘 경기 0 개의 땅볼 아웃으로 타자 잡아냈습니다.
                        투수 오늘 경기 0 개의 싱글 안타 허용하였습니다.
                        투수 오늘 경기 0 개의 2루타 허용하였습니다.

                    투수 이번 시즌 0의 평균 자책점을 기록하고 있습니다.
                    투수 저번 타석 0을 기록하였습니다.

                투수 과연 어떤 공을 던질까요?
            """
            query = "SELECT ?o WHERE {?pitcher ?thisERA ?o}"
            r = self.rdf.query(query, initBindings={"pitcher": pitcher, "thisERA": self.thisERA})
            era = list(r)[0][0]

            query = "SELECT ?o WHERE {?s ?fromPitcher ?pitcher . ?s ?inGame ?thisGame . ?s ?result ?o} order by ?s"
            r = self.rdf.query(query, initBindings={"fromPitcher": self.fromPitcher, "pitcher": pitcher,
                                                  "inGame": self.inGame, "thisGame": thisGame,
                                                  "result": self.result})

            total_batterbox = len(list(r)) + 1
            strikeout = len([1 for i in r if 'Strikeout' in i[0]])
            baseonballs = len([1 for i in r if 'BaseOnBalls' in i[0]])
            fly = len([1 for i in r if 'Fly' in i[0]])
            outinbase = len([1 for i in r if 'OutInBase' in i[0]])
            singlehit = len([1 for i in r if 'SingleHit' in i[0]])
            double = len([1 for i in r if 'Double' in i[0]])

            if(r):
                recent_result = self.change_result_history_to_korean(list(r)[-1][0].split("#")[1].split("_")[1])

            annotation_about_this_game = ["투수 오늘 경기 "+str(total_batterbox)+"번째 타석에서 공을 던집니다",
                                          "투수 오늘 경기 "+str(total_batterbox)+"번째 타자를 상대하고 있습니다",
                                          "투수 오늘 경기 "+str(strikeout)+"개의 삼진을 잡아내고 있습니다",
                                          "투수 오늘 경기 "+str(baseonballs)+"개의 포볼로 타자 출루 시켰습니다",
                                          "투수 오늘 경기 "+str(fly)+"개의 플라이 아웃으로 타자 잡아냈습니다",
                                          "투수 오늘 경기 "+str(outinbase)+"개의 땅볼 아웃으로 타자 잡아냈습니다",
                                          "투수 오늘 경기 "+str(singlehit)+"개의 싱글 안타 허용하였습니다",
                                          "투수 오늘 경기 "+str(double)+"개의 2루타 허용하였습니다",

                                          pitcher_name + " 투수 오늘 경기 " + str(total_batterbox) + "번째 타석에서 공을 던집니다",
                                          pitcher_name + " 투수 오늘 경기 " + str(total_batterbox) + "번째 타자를 상대하고 있습니다",
                                          pitcher_name + " 투수 오늘 경기 " + str(strikeout) + "개의 삼진을 잡아내고 있습니다",
                                          pitcher_name + " 투수 오늘 경기 " + str(baseonballs) + "개의 포볼로 타자 출루 시켰습니다",
                                          pitcher_name + " 투수 오늘 경기 " + str(fly) + "개의 플라이 아웃으로 타자 잡아냈습니다",
                                          pitcher_name + " 투수 오늘 경기 " + str(outinbase) + "개의 땅볼 아웃으로 타자 잡아냈습니다",
                                          pitcher_name + " 투수 오늘 경기 " + str(singlehit) + "개의 싱글 안타 허용하였습니다",
                                          pitcher_name + " 투수 오늘 경기 " + str(double) + "개의 2루타 허용하였습니다",

                                          pitcher_name + " 오늘 경기 " + str(total_batterbox) + "번째 타석에서 공을 던집니다",
                                          pitcher_name + " 오늘 경기 " + str(total_batterbox) + "번째 타자를 상대하고 있습니다",
                                          pitcher_name + " 오늘 경기 " + str(strikeout) + "개의 삼진을 잡아내고 있습니다",
                                          pitcher_name + " 오늘 경기 " + str(baseonballs) + "개의 포볼로 타자 출루 시켰습니다",
                                          pitcher_name + " 오늘 경기 " + str(fly) + "개의 플라이 아웃으로 타자 잡아냈습니다",
                                          pitcher_name + " 오늘 경기 " + str(outinbase) + "개의 땅볼 아웃으로 타자 잡아냈습니다",
                                          pitcher_name + " 오늘 경기 " + str(singlehit) + "개의 싱글 안타 허용하였습니다",
                                          pitcher_name + " 오늘 경기 " + str(double) + "개의 2루타 허용하였습니다",
                                          ]
            annotation = annotation + list(map("".join, product(situation, annotation_about_this_game)))

            if(total_batterbox > 1):
                annotation = annotation + ["투수 지난 타석 " + str(recent_result) + "을 기록하였습니다",
                                           pitcher_name + " 투수 지난 타석 " + str(recent_result) + "을 기록하였습니다",
                                           pitcher_name + " 지난 타석 " + str(recent_result)+ "을 기록하였습니다",
                                           ]

            annotation = annotation + ["투수 이번 시즌 " + str(era) + "의 평균 자책점을 기록하고 있습니다",
                                       pitcher_name + " 투수 이번 시즌 " + str(era) + "의 평균 자책점을 기록하고 있습니다",
                                       pitcher_name + " 이번 시즌 " + str(era) + "의 평균 자책점을 기록하고 있습니다",

                                       pitcher_name + " 투수",
                                       pitcher_name + " 투수 어떤 공을 던질까요",
                                       ]

        if(isabouthitter):
            """
            타자 오늘 0번째 타석 입니다
            타자 오늘 0번째 타석에서 섰습니다
                if 타석 > 1
                타자 오늘 0번째 타석에서 0개의 안타 기록했습니다
                타자 오늘 0개의 안타 기록합니다
                타자 저번 타석 0을 기록하였습니다
                    if 아웃 >= 1
                    타자 오늘 0번째 타석에서 0개의 0아웃 기록했습니다
    
    
            타자 이번 시즌 0의 평균 타율을 기록하고 있습니다
            타자 이번 타석 안타를 기록 할 수 있을까요
            """

            query = "SELECT ?o where {?batter ?thisAVG ?o}"
            r = self.rdf.query(query, initBindings={"batter": hitter, "thisAVG": self.thisAVG})
            avg = list(r)[0][0]

            query = "SELECT ?o where {?s ?toHitter ?hitter . ?s ?inGame ?thisGame . ?s ?result ?o } order by ?s"
            r = self.rdf.query(query, initBindings={"toHitter": self.toHitter, "hitter": hitter,
                                                    "inGame": self.inGame, "thisGame": thisGame, "result": self.result})

            total_batterbox = len(list(r)) + 1
            strikeout = len([1 for i in r if 'Strikeout' in i[0]])
            baseonballs = len([1 for i in r if 'BaseOnBalls' in i[0]])
            fly = len([1 for i in r if 'Fly' in i[0]])
            outinbase = len([1 for i in r if 'OutInBase' in i[0]])
            singlehit = len([1 for i in r if 'SingleHit' in i[0]])
            double = len([1 for i in r if 'Double' in i[0]])
            triple = len([1 for i in r if 'Triple' in i[0]])
            homerun = len([1 for i in r if 'HomeRun' in i[0]])

            hits = int(singlehit) + int(double) + int(triple) + int(homerun)
            outs = int(fly) + int(outinbase) + int(strikeout)

            if(r):
                recent_result = self.change_result_history_to_korean(list(r)[-1][0].split("#")[1].split("_")[1])

            annotation_about_this_game = ["타자 오늘 경기 " + str(total_batterbox) + "번째 타석입니다",
                                          "타자 오늘 경기 " + str(total_batterbox) + "번째 타석에 섰습니다",

                                          hitter_name + " 타자 오늘 경기 " + str(total_batterbox) + "번째 타석입니다",
                                          hitter_name + " 타자 오늘 경기 " + str(total_batterbox) + "번째 타석에 섰습니다",

                                          hitter_name + " 오늘 경기 " + str(total_batterbox) + "번째 타석입니다",
                                          hitter_name + " 오늘 경기 " + str(total_batterbox) + "번째 타석에 섰습니다",
                                          ]
            if(total_batterbox > 1):
                annotation_about_this_game = annotation_about_this_game + [
                    "타자 오늘 " + str(total_batterbox) + "번째 타석에서 " + str(hits) + "개의 안타 기록했습니다",
                    "타자 오늘 " + str(singlehit) + "개의 1루타 기록했습니다",
                    "타자 오늘 " + str(double) + "개의 2루타 기록했습니다",
                    "타자 오늘 " + str(triple) + "개의 3루타 기록했습니다",
                    "타자 저번 타석 " + str(recent_result) + "을 기록하였습니다",

                    hitter_name + " 타자 오늘 " + str(total_batterbox) + "번째 타석에서 " + str(hits) + "개의 안타 기록했습니다",
                    hitter_name + " 타자 오늘 " + str(singlehit) + "개의 1루타 기록했습니다",
                    hitter_name + " 타자 오늘 " + str(double) + "개의 2루타 기록했습니다",
                    hitter_name + " 타자 오늘 " + str(triple) + "개의 3루타 기록했습니다",
                    hitter_name + " 타자 저번 타석 " + str(recent_result) + "을 기록하였습니다",

                    hitter_name + " 오늘 " + str(total_batterbox) + "번째 타석에서 " + str(hits) + "개의 안타 기록했습니다",
                    hitter_name + " 오늘 " + str(singlehit) + "개의 1루타 기록했습니다",
                    hitter_name + " 오늘 " + str(double) + "개의 2루타 기록했습니다",
                    hitter_name + " 오늘 " + str(triple) + "개의 3루타 기록했습니다",
                    hitter_name + " 저번 타석 " + str(recent_result) + "을 기록하였습니다",
                ]
            if(outs > 0):
                annotation_about_this_game = annotation_about_this_game + [
                    "타자 오늘 " + str(total_batterbox) + "번째 타석에서 " + str(outs) + "개의 아웃 기록했습니다",
                    "타자 오늘 " + str(outs) + "개의 아웃 기록했습니다",
                    "타자 오늘 " + str(fly) + "개의 플라이 아웃 기록했습니다",
                    "타자 오늘 " + str(outinbase) + "개의 땅볼 아웃 기록했습니다",

                    hitter_name + " 타자 오늘 " + str(total_batterbox) + "번째 타석에서 " + str(outs) + "개의 아웃 기록했습니다",
                    hitter_name + " 타자 오늘 " + str(outs) + "개의 아웃 기록했습니다",
                    hitter_name + " 타자 오늘 " + str(fly) + "개의 플라이 아웃 기록했습니다",
                    hitter_name + " 타자 오늘 " + str(outinbase) + "개의 땅볼 아웃 기록했습니다",

                    hitter_name + " 오늘 " + str(total_batterbox) + "번째 타석에서 " + str(outs) + "개의 아웃 기록했습니다",
                    hitter_name + " 오늘 " + str(outs) + "개의 아웃 기록했습니다",
                    hitter_name + " 오늘 " + str(fly) + "개의 플라이 아웃 기록했습니다",
                    hitter_name + " 오늘 " + str(outinbase) + "개의 땅볼 아웃 기록했습니다",
                ]
            if(strikeout > 0):
                annotation_about_this_game = annotation_about_this_game + [
                    "타자 오늘 " + str(total_batterbox) + "번째 타석에서 " + str(strikeout) + "개의 삼진 아웃 당했습니다",
                    "타자 오늘 " + str(strikeout) + "개의 삼진 아웃 당헀습니다",

                    hitter_name + " 타자 오늘 " + str(total_batterbox) + "번째 타석에서 " + str(strikeout) + "개의 삼진 아웃 당했습니다",
                    hitter_name + " 타자 오늘 " + str(strikeout) + "개의 삼진 아웃 당헀습니다",

                    hitter_name + " 오늘 " + str(total_batterbox) + "번째 타석에서 " + str(strikeout) + "개의 삼진 아웃 당했습니다",
                    hitter_name + " 오늘 " + str(strikeout) + "개의 삼진 아웃 당헀습니다",
                ]
            if(baseonballs > 0):
                annotation_about_this_game = annotation_about_this_game + [
                    "타자 오늘 " + str(total_batterbox) + "번째 타석에서 " + str(baseonballs) + "개의 포볼로 출루 하였습니다",
                    "타자 오늘 " + str(baseonballs) + "개의 포볼 기록합니다",

                    hitter_name + " 타자 오늘 " + str(total_batterbox) + "번째 타석에서 " + str(baseonballs) + "개의 포볼로 출루 하였습니다",
                    hitter_name + " 타자 오늘 " + str(baseonballs) + "개의 포볼 기록합니다",

                    hitter_name + " 오늘 " + str(total_batterbox) + "번째 타석에서 " + str(baseonballs) + "개의 포볼로 출루 하였습니다",
                    hitter_name + " 오늘 " + str(baseonballs) + "개의 포볼 기록합니다",
                ]
            if(homerun > 0):
                annotation_about_this_game = annotation_about_this_game + [
                    "타자 오늘 홈런 기록하였습니다",
                    hitter_name + " 타자 오늘 홈런 기록하였습니다",
                    hitter_name + " 오늘 홈런 기록하였습니다",
                ]

            annotation = annotation + list(map("".join, product(situation, annotation_about_this_game)))
            annotation = annotation + ["타자 이번 시즌 " + str(avg) + "의 평균 타율을 기록하고 있습니다",
                                       hitter_name + " 타자 이번 시즌 " + str(avg) + "의 평균 타율을 기록하고 있습니다",
                                       hitter_name + " 이번 시즌 " + str(avg) + "의 평균 타율을 기록하고 있습니다",

                                       hitter_name + " 타자",
                                       ]

        if(isaboutpitcher and isabouthitter):
            query = "SELECT ?o where {?s ?inGame ?thisGame . ?s ?toHitter ?hitter . ?s ?fromPitcher ?pitcher . ?s ?result ?o} order by desc(?s)"
            r = self.rdf.query(query, initBindings={
                "inGame": self.inGame, "thisGame": thisGame,
                "toHitter": self.toHitter, "hitter": hitter,
                "fromPitcher": self.fromPitcher, "pitcher": pitcher,
                "result": self.result
            })

            total_batterbox = len(list(r)) + 1
            strikeout = len([1 for i in r if 'Strikeout' in i[0]])
            baseonballs = len([1 for i in r if 'BaseOnBalls' in i[0]])
            fly = len([1 for i in r if 'Fly' in i[0]])
            outinbase = len([1 for i in r if 'OutInBase' in i[0]])
            singlehit = len([1 for i in r if 'SingleHit' in i[0]])
            double = len([1 for i in r if 'Double' in i[0]])
            triple = len([1 for i in r if 'Triple' in i[0]])
            homerun = len([1 for i in r if 'HomeRun' in i[0]])

            if(r):
                history = [self.change_result_history_to_korean(row[0].split("#")[1].split("_")[1]) for row in r]
                recent_result = history[0]

            hits = int(singlehit) + int(double) + int(triple) + int(homerun)
            outs = int(fly) + int(outinbase)

            annotation = annotation + [hitter_name + " 타자 " + pitcher_name + " 투수를 상대로 오늘 " + str(hits) + "개의 안타 기록 하였습니다",
                                       pitcher_name + " 투수 " + hitter_name + " 타자를 상대로 오늘 경기 " + str(hits) + "개의 안타를 허용 하였습니다",
                                       pitcher_name + " 투수 " + hitter_name + " 타자를 상대로 오늘 경기 " + str(total_batterbox) + "번째 대결입니다",
                                       "투수와 타자 사이에 팽팽한 긴장감이 감지됩니다.",
                                       ]
            if (strikeout > 0):
                annotation = annotation + [pitcher_name + " 투수 " + hitter_name + " 타자를 상대로 오늘 경기 " + str(strikeout) + "개의 스트라이크 아웃을 잡아냈습니다",
                                           hitter_name + " 타자 " + pitcher_name + " 투수 상대로 오늘 경기 " + str(strikeout) + "개의 스트라이크 아웃 당했습니다",
                                           ]
            if (fly > 0):
                annotation = annotation + [pitcher_name + " 투수 " + hitter_name + " 타자를 상대로 오늘 경기 " + str(fly) + "개의 플라이 아웃을 잡아냈습니다",
                                           hitter_name + " 타자 " + pitcher_name + " 투수 상대로 오늘 경기 " + str(fly) + "개의 플라이 아웃 당했습니다",
                                           ]
            if (baseonballs > 0):
                annotation = annotation + [pitcher_name + " 투수 " + hitter_name + " 타자를 상대로 오늘 경기 " + str(baseonballs) + "개의 포볼로 출루 시켰습니다",
                                           hitter_name + " 타자 " + pitcher_name + " 투수 상대로 오늘 경기 " + str(baseonballs) + "개의 포볼로 출루 하였습니다",
                                           ]
            if (homerun > 0):
                annotation = annotation + [pitcher_name + " 투수 " + hitter_name + " 타자를 상대로 오늘 경기 홈런을 허용하였습니다",
                                           hitter_name + " 타자 " + pitcher_name + " 투수 상대로 오늘 경기 홈런 기록하였습니다",
                                           ]
            if(total_batterbox > 1):
                annotation = annotation + [
                    pitcher_name + " 투수 " + hitter_name + " 타자를 상대로 저번 타석 " + str(recent_result) + " 기록하였습니다",
                    hitter_name + " 타자 " + pitcher_name + " 투수 상대로 저번 타석 " + str(recent_result) + " 기록하였습니다",
                    ]

        if(isaboutrunner):
            """
            각 루에 누가 있는지
                ex) 0루에 OOO(이)가 있습니다.
                ex) 0루에 OOO(이)가 주자로 있습니다.
                ex) OOO(이)가 나가있습니다.
                ex) 000(이)가 득점권에 있습니다.

            1루에 주자 있을 때
                ex) 타자 1루 주자 있었던 최근 타석 000를 기록하였습니다.
                ex) 오늘 1루 주자가 있는 타석에서 타자 000를 기록하였습니다.
                ex) 투수 1루 주자 있었던 최근 타석 000를 기록하였습니다.
            """
            first_runner, second_runner, third_runner = self.search_runner(batterbox=batterbox)
            if(first_runner):
                first_runner = self.get_player_name(first_runner.split("#")[-1])
            if(second_runner):
                second_runner = self.get_player_name(second_runner.split("#")[-1])
            if(third_runner):
                third_runner = self.get_player_name(third_runner.split("#")[-1])

            if(first_runner or second_runner or third_runner):
                annotation = annotation + [" ".join([i for i in [first_runner, second_runner, third_runner] if i is not None]) + " 주자로 나가있습니다",
                                           "주자에는 " + " ".join([i for i in [first_runner, second_runner, third_runner] if i is not None]) + "가 있습니다"
                                           ]
            if(first_runner):
                query = "SELECT ?o where {?s ?toHitter ?hitter . ?s ?result ?o . ?s ?stayIn1stBase ?o1} order by ?s"
                r = self.rdf.query(query, initBindings={"toHitter": self.toHitter, "hitter": hitter,
                                                      "inGame": self.inGame, "thisGame": thisGame,
                                                      "result": self.result, "stayIn1stBase": self.stayIn1stBase})
                if(r):
                    recent_result = self.change_result_history_to_korean(list(r)[-1][0].split("#")[1].split("_")[1])

                    annotation = annotation + ["타자 1루 주자가 있는 타석에서 최근 " + str(recent_result) + "을 기록하였습니다",
                                               hitter_name + " 타자 1루 주자가 있는 타석에서 최근 " + str(recent_result) + "을 기록하였습니다"
                                               ]

                query = "SELECT ?o where {?s ?fromPitcher ?pitcher . ?s ?result ?o . ?s ?stayIn1stBase ?o1} order by ?s"
                r = self.rdf.query(query, initBindings={"fromPitcher": self.fromPitcher, "pitcher": pitcher,
                                                        "inGame": self.inGame, "thisGame": thisGame,
                                                        "result": self.result, "stayIn1stBase": self.stayIn1stBase})

                if(r):
                    recent_result = self.change_result_history_to_korean(list(r)[-1][0].split("#")[1].split("_")[1])

                    annotation = annotation + ["투수 1루 주자가 있는 타석에서 최근 " + str(recent_result) + "을 기록하였습니다",
                                               pitcher_name + " 투수 1루 주자가 있는 타석에서 최근 " + str(recent_result) + "을 기록하였습니다"
                                               ]

                annotation = annotation + ["1루에는 " + str(first_runner) + "가 주자로 있습니다",
                                           "1루에는 " + str(first_runner) + "가 있습니다",
                                           str(first_runner) + " 선수 1루에 있습니다",
                                           ]
            if(second_runner):
                annotation = annotation + ["득점권에 주자 나가 있습니다",
                                           str(second_runner) + ", 득점권에 주자로 있습니다",
                                           str(second_runner) + " 선수 2루에 있습니다",
                                           "2루에는 " + str(second_runner) + "가 주자로 있습니다",
                                           "2루에는 " + str(second_runner) + "가 있습니다",
                                           ]
            if(third_runner):
                annotation = annotation + ["득점권에 주자 나가 있습니다",
                                           str(third_runner) + ", 득점권에 주자로 있습니다",
                                           str(third_runner) + " 선수 3루에 있습니다",
                                           "3루에는 " + str(third_runner) + "가 주자로 있습니다",
                                           "3루에는 " + str(third_runner) + "가 있습니다",
                                           ]

        return annotation

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
        stadium = gameinfo[0]
        home_Fname = gameinfo[3][0]
        away_Fname = gameinfo[3][1]

        annotation = []
        if(btop == 0): #away defense
            new = [home_Fname + "팀 공격을 하고 있습니다.",
                   "홈팀 공격하고 있습니다.",
                   stadium + "에서 " + home_Fname + "팀 공격을 하고 있습니다.",
                   stadium + "에서 " + "홈팀 공격하고 있습니다."
                   ]

        else:
            new = [away_Fname + "팀 공격을 하고 있습니다.",
                   "어웨이팀 공격하고 있습니다.",
                   stadium + "에서 " + away_Fname + "팀 공격을 하고 있습니다.",
                   stadium + "에서 " + "어웨이팀 공격하고 있습니다."
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

        annotation = [
            str(stadium) + "에서 경기 진행중입니다.",
            "현재 경기 스코어 " + str(homescore) + " 대 " + str(awayscore) + " 입니다.",
            str(inn).split("_")[-1] + " 경기 스코어 " + str(homescore) + " 대 " + str(awayscore) + " 진행중입니다.",
            home_full_name + " 대 " + away_full_name + " 현재 " + str(homescore) + " 대 " + str(awayscore) + " 진행중입니다.",
        ]

        annotation = annotation + [about_atmosphere + " " + i for i in annotation]

        return annotation

    def get_motion_annotation(self, scene_label, motion_label, who=None, resource=None):
        anno = []
        if (who):
            if(who == "pitcher" and motion_label == 3):
                pitcher = self.get_player_name(resource.get_pitcher())
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
                player = self.get_player_name(player)
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

    def change_result_history_to_korean(self, h):
        result = ["BaseByError", "BaseOnBalls", "HitByPitch", "Double", "HomeRun", "Triple", "SingleHit", "Fly", "OutInBase", "Strikeout", "Out", "GetOnBase"]
        korean = ["실책 출루", "포 볼", "데드 볼", "2루타", "홈런", "3루타", "1루타", "플라이 아웃", "땅볼 아웃", "스트라이크 아웃", "아웃", "출루"]

        return korean[result.index(h)]

    def get_player_name(self, name):
        return "".join([s for s in list(name) if not s.isdigit()])

