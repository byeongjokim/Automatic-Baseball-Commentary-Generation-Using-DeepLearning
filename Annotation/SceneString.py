import random
import rdflib

#######################################################################################################BatterBox
def BatterBox(gamecode, b, p):
    g = rdflib.Graph()
    g.load('_data/_owl/180515SKOB.owl')
    uri = "http://ailab.hanyang.ac.kr/ontology/baseball#"

    thisAVG = rdflib.URIRef(uri + "thisAVG")
    thisERA = rdflib.URIRef(uri + "thisERA")

    toHitter = rdflib.URIRef(uri + "toHitter")
    fromPitcher = rdflib.URIRef(uri + "fromPitcher")
    inGame = rdflib.URIRef(uri + "inGame")
    thisGame = rdflib.URIRef(uri + gamecode)
    result = rdflib.URIRef(uri + "result")


    batter = rdflib.URIRef(uri + b)
    pitcher = rdflib.URIRef(uri + p)

    query = "SELECT ?o where {?s ?p ?o}"
    r = g.query(query, initBindings={"s":batter, "p": thisAVG})
    for row in r:
        avg = row[0]

    r = g.query(query, initBindings={"s": pitcher, "p": thisERA})
    for row in r:
        era = row[0]

    query = "SELECT ?s ?o where {?s ?toHitter ?Hitter . ?s ?inGame ?thisGame . ?s ?result ?o}"
    r = g.query(query, initBindings={"toHitter":toHitter, "Hitter":batter, "inGame":inGame, "thisGame":thisGame})
    this_game_count = len(r)

    batter_history = []
    for row in r:
        batter_history.append(row[0].split("#")[1].split("_")[1])



    query = "SELECT ?o where {?s ?fromPitcher ?pitcher . ?s ?inGame ?thisGame . ?s ?result ?o}"
    r = g.query(query, initBindings={"fromPitcher": fromPitcher, "pitcher": pitcher,
                                     "inGame": inGame, "thisGame": thisGame,
                                     "result": result})
    total_out = len(r)
    strikeout = 0
    fourball = 0
    for row in r:
        if ("Strikeout" in row[0]):
            strikeout = strikeout + 1
        if ("BaseOnBalls" in row[0]):
            fourball = fourball + 1

    annotation = [
        b + " 타자와 " + p + " 투수의 신경전 속에 " + b + " 타자는 이번시즌 " + str(avg) + "의 평균 타율을 기록하고 있습니다.",
        b + " 타자와 " + p + " 투수의 신경전 속에 " + p + " 투수는 이번시즌 " + str(era) + "의 평균 자책점을 기록하고 있습니다.",

        b + " 타자의 오늘 " + str(this_game_count) + " 번째 타석입니다.",
        b + " 타자 오늘 " + str(this_game_count) +" 번째 타석, " + ", ".join(_ for _ in batter_history) + "을 기록하고 있습니다.",
        b + " 타자는 이번시즌 " + str(avg) + "의 평균 타율을 기록하고 있습니다.",

        p + " 투수 오늘 경기 " + str(total_out) + " 개의 아웃을 잡아내고 있습니다.",
        p + " 투수 오늘 경기 " + str(strikeout) + " 개의 삼진을 잡아내고 있습니다.",
        p + " 투수 오늘 경기 " + str(fourball) + " 개의 포볼로 타자를 진루 시켰습니다.",
    ]
    return random.choice(annotation)

'''
def coach():
    annotation = [
        "코치 모습이 보이네요. 지고있는 상황에 어떤 전술을 쓸까요?",
        "삼성 코치진 모습이 보입니다."
    ]
    return random.choice(annotation)

def gallery():
    annotation = [
        "(날씨)날씨가 ~~ 함에도 불구하고 많은 관객분들이 찾아 주셨습니다.",
        "관중들의 열띤 응원속에 경기가 달아오르고 있습니다."
    ]
    return random.choice(annotation)

def OutField(p):
    pos = "외야"
    if(p == "left"):
        pos = "좌익수"
    elif(p == "right"):
        pos = "우익수"
    elif (p == "center"):
        pos = "중간"

    annotation = [
        pos + "쪽의 ~~ 선수 모습이 보입니다."
    ]
    return random.choice(annotation)

def Base(pos):
    annotation = [
        pos + "루 주자 ~~~ ",
        pos + "루수 선수 ~~~ "
    ]

    return random.choice(annotation) + Player("p")

def Player(player):
    if(player == "ss"):
        annotation = [
            "유격수의 모습이 보이네요."
        ]

    else:
        annotation = [
            "이 선수는 ~~~~ 인 선수 입니다."
        ]

    return random.choice(annotation)

def etc():
    annotation = [
        "기타장면 입니다."
    ]
    return random.choice(annotation)
'''