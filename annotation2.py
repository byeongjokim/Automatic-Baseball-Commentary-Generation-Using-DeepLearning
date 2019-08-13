import rdflib
import settings
from itertools import product

class Annotation():
    def __init__(self):
        self.rdf = rdflib.Graph()
        #self.rdf.load(settings.OWL_FILE)
        self.rdf.load("./_data/owl/baseball.owl")

        #self.uri = settings.OWL_URI
        self.uri = "http://ailab.hanyang.ac.kr/ontology/baseball#"

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
            inn + " " + str(sbo[0]) + " 스트라이크 " + str(sbo[1]) + " 볼 " + str(sbo[2]) + " 아웃 상황, ",
            inn + " 진행중에, ",
            str(sbo[0]) + " 스트라이크 " + str(sbo[1]) + " 볼 " + str(sbo[2]) + " 아웃 상황, ",
            str(sbo[0]) + " 스트라이크 " + str(sbo[1]) + " 볼 상황, ",
            str(sbo[2]) + " 아웃 상황, ",

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

    def about_batterbox(self, gameCode="20180906LGNC", gameinfo=["잠실", "20180906", ["NC", "LT"], ["NC 다이노소어", "롯데 어쩌고"]], inn="20180906LGNC_3회말", score=[0, 2], batterbox="baseball.20180906LGNC_BatterBox_026", sbo=[2, 1, 2], isaboutpitcher="왕웨이중68948", isaboutbatter="유강남61102", isaboutrunner=None):
        bbox = str(batterbox).split(".")[-1]
        batterbox_uri = rdflib.URIRef(self.uri + bbox)
        thisGame = rdflib.URIRef(self.uri + gameCode)
        pitcher = rdflib.URIRef(self.uri + isaboutpitcher)
        pitcher_name = self.get_player_name(isaboutpitcher)
        hitter = rdflib.URIRef(self.uri + isaboutbatter)
        hitter_name = self.get_player_name(isaboutbatter)

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

            total_batterbox = len(list(r))
            strikeout = len([1 for i in r if 'Strikeout' in i[0]])
            baseonballs = len([1 for i in r if 'BaseOnBalls' in i[0]])
            fly = len([1 for i in r if 'Fly' in i[0]])
            outinbase = len([1 for i in r if 'OutInBase' in i[0]])
            singlehit = len([1 for i in r if 'SingleHit' in i[0]])
            double = len([1 for i in r if 'Double' in i[0]])

            recent_result = self.change_result_history_to_korean(list(r)[0][0].split("#")[1].split("_")[1])

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
            annotation = annotation + ["투수 이번 시즌 " + str(era) + "의 평균 자책점을 기록하고 있습니다",
                                       pitcher_name + " 투수 이번 시즌 " + str(era) + "의 평균 자책점을 기록하고 있습니다",
                                       pitcher_name + " 이번 시즌 " + str(era) + "의 평균 자책점을 기록하고 있습니다",
                                       "투수 지난 타석 " + str(recent_result) + "을 기록하였습니다",
                                       pitcher_name + " 투수 지난 타석 " + str(recent_result) + "을 기록하였습니다",
                                       pitcher_name + " 지난 타석 " + str(recent_result)+ "을 기록하였습니다",
                                       ]

        # if(isaboutbatter):
        #     annotation.append(1)
        #
        # if(isaboutpitcher and isaboutbatter):
        #     annotation.append(1)
        #
        # if(isaboutrunner):
        #     runner_uri = self.search_runner(batterbox=batterbox)
        #
        #     """
        #     각 루에 누가 있는지
        #         ex) 0루에 OOO(이)가 있습니다.
        #         ex) 0루에 OOO(이)가 주자로 있습니다.
        #         ex) OOO(이)가 나가있습니다.
        #         ex) 000(이)가 득점권에 있습니다.
        #
        #     1루에 주자 있을 때
        #         ex) 타자 1루 주자 있었던 최근 타석 000를 기록하였습니다.
        #         ex) 오늘 1루 주자가 있는 타석에서 타자 000를 기록하였습니다.
        #         ex) 투수 1루 주자 있었던 최근 타석 000를 기록하였습니다.
        #     """

        return annotation

    def change_result_history_to_korean(self, h):
        result = ["BaseByError", "BaseOnBalls", "HitByPitch", "Double", "HomeRun", "Triple", "SingleHit", "Fly", "OutInBase", "Strikeout", "Out", "GetOnBase"]
        korean = ["실책 출루", "포 볼", "데드 볼", "2루타", "홈런", "3루타", "1루타", "플라이 아웃", "땅볼 아웃", "스트라이크 아웃", "아웃", "출루"]

        return korean[result.index(h)]

    def get_player_name(self, name):
        return "".join([s for s in list(name) if not s.isdigit()])

a = Annotation()
print(a.about_batterbox())