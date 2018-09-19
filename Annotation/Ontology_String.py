import rdflib

class Ontology_String():
    def __init__(self):
        self.g = rdflib.Graph()
        #self.g.load('../_data/_owl/180515SKOB.owl')
        self.g.load('_data/_owl/baseball.owl')
        self.uri = "http://ailab.hanyang.ac.kr/ontology/baseball#"
        self.set_object_properties()
        self.set_data_properties()

    def set_instance(self):

        return 1

    def set_object_properties(self):
        #arrivedTo1stBase = rdflib.URIRef(self.uri + "arrivedTo1stBase")

        self.inGame = rdflib.URIRef(self.uri + "inGame")

        self.fromPitcher = rdflib.URIRef(self.uri + "fromPitcher")
        self.toHitter = rdflib.URIRef(self.uri + "toHitter")

        self.stayIn1stBase = rdflib.URIRef(self.uri + "stayIn1stBase")
        self.stayIn2ndBase = rdflib.URIRef(self.uri + "stayIn2ndBase")
        self.stayIn3rdBase = rdflib.URIRef(self.uri + "stayIn3rdBase")

        self.result = rdflib.URIRef(self.uri + "result")

    def set_data_properties(self):
        self.thisERA = rdflib.URIRef(self.uri + "thisERA")
        self.thisAVG = rdflib.URIRef(self.uri + "thisAVG")

    def search_gameInfo(self, gameCode, inn, score, gameinfo):
        homescore = score[0]
        awayscore = score[1]

        stadium = gameinfo[0]
        date = gameinfo[1]

        annotation = [
            str(date) + " " + str(stadium) + "에서 경기 진행중입니다.",
            str(date) + "에 진행하는 경기 현재 경기 스코어 " + str(homescore) + " 대 " + str(awayscore) + " 입니다.",
            # str(inn).split("_")[-1] + " 경기 스코어 "+str(homescore)+" 대 "+str(awayscore)+" 입니다.",
            str(inn).split("_")[-1] + " 경기 스코어 " + str(homescore) + " 대 " + str(awayscore) + " 진행중입니다."
        ]
        return annotation

    def search_pitcher(self, gameCode, p):
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

        for row in r:
            if ("Strikeout" in row[0]):
                strikeout = strikeout + 1
            if ("BaseOnBalls" in row[0]):
                baseonballs = baseonballs + 1

        annotation = [
            p + " 투수 오늘 경기 " + str(len(r) + 1) + "번째 타석에서 공을 던지고 있습니다.",
            p + " 투수 오늘 경기 " + str(strikeout) + "개의 삼진을 잡아내고 있습니다.",
            p + " 투수 오늘 경기 " + str(baseonballs) + "개의 포볼로 타자를 진루 시켰습니다.",
            p + " 투수 이번 시즌 " + str(era) + "의 평균 자책점을 기록하고 있습니다.",
            p + " 투수 과연 어떤 공을 던질까요?",
        ]

        return annotation

    def search_batter(self, gameCode, b):
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
            batter_history.append(row[0].split("#")[1].split("_")[1])

        query = "SELECT ?o where {?s ?toHitter ?hitter . ?s ?result ?o . ?s ?stayIn1stBase ?o1} order by ?s"
        r = self.g.query(query, initBindings={"toHitter": self.toHitter, "hitter": batter,
                                         "inGame": self.inGame, "thisGame": thisGame,
                                         "result": self.result, "stayIn1stBase": self.stayIn1stBase})

        batter_history_when1st = []
        for row in r:
            batter_history_when1st.append(row[0].split("#")[1].split("_")[1])

        annotation = [
            b + " 타자의 오늘 " + str(this_game_count + 1) + "번째 타석입니다.",
            b + " 타자는 이번 시즌 " + str(avg) + "의 평균 타율을 기록하고 있습니다.",
            b + " 타자 이번 타석 안타를 기록 할 수 있을까요?"
        ]

        if (batter_history):
            annotation.append(b + " 타자 오늘 " + str(this_game_count + 1) + "번째 타석입니다, " + ", ".join(_ for _ in batter_history) + "을 기록하고 있습니다.")
            annotation.append(b + " 타자 저번 타석, " + str(batter_history[-1]) + "을 기록하였습니다.")
        if (batter_history_when1st):
            annotation.append(b + " 타자 1루 주자가 있는 상황에서 " + ", ".join(_ for _ in batter_history_when1st) + "을 기록하고 있습니다.")
            annotation.append("오늘 1루 주자가 있는 타석에서 " + b + " 타자 최근, " + str(batter_history_when1st[-1]) + "을 기록하였습니다.")

        return annotation

    def search_pitcherbatter(self, gameCode, p, b):
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
            result_history.append(row[0].split("#")[1].split("_")[1])

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

        annotation = [
            b + " 타자 " + p + " 투수의 신경전 속에 " + b + " 타자는 이번시즌 " + str(avg) + "의 펑균 타율을 기록하고 있습니다.",
            b + " 타자 " + p + " 투수의 신경전 속에 " + p + " 투수는 이번시즌 " + str(era) + "의 평균 자책점을 기록하고 있습니다.",
            b + " 타자 " + p + " 투수를 상대로 오늘 " + ", ".join(_ for _ in result_history) + "을 기록 하였습니다.",
            b + " 타자 " + p + " 투수를 상대로 오늘 " + str(getonbase) + "개의 안타 기록 하였습니다.",
            p + " 투수 " + b + " 타자를 상대로 오늘 경기 " + str(getonbase) + "개의 안타를 허용 하였습니다.",
            p + " 투수 " + b + " 타자를 상대로 오늘 경기 " + str(strikeout) + "개의 스트라이크 아웃을 잡아냈습니다.",
            "투수와 타자 사이에 팽팽한 긴장감이 감지됩니다.",
        ]

        return annotation

    def search_runner(self, batterbox):
        annotation = []
        first_runner_annotation, first_runner = self.search_first_runner(batterbox)
        second_runner_annotation, second_runner = self.search_second_runner(batterbox)
        third_runner_annotation, third_runner = self.search_third_runner(batterbox)

        annotation = annotation + first_runner_annotation
        annotation = annotation + second_runner_annotation
        annotation = annotation + third_runner_annotation

        result = ""
        if(first_runner):
            result = result + "1루에는 " + str(first_runner) + " "
        if (second_runner):
            result = result + "2루에는 " + str(second_runner) + " "
        if (third_runner):
            result = result + "3루에는 " + str(third_runner) + " "

        result = result + "(이)가 나가있습니다."

        annotation.append(result)

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
            annotation = [
                "1루에 " + str(runner) + "선수가 주자로 나와있습니다.",
                "1루에 " + str(runner) + "선수가 나가 있습니다.",
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
            annotation = [
                "2루에 " + str(runner) + "선수가 주자로 나와있습니다.",
                "2루에 " + str(runner) + "선수가 나가 있습니다.",
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
            annotation = [
                "3루에 " + str(runner) + "선수가 주자로 나와있습니다.",
                "3루에 " + str(runner) + "선수가 나가 있습니다.",
            ]

        return annotation, runner