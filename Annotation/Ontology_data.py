import owlready2
import re
import rdflib

def create_game(onto, GameInfo):

    game = onto.Game(GameInfo["DateHomeAway"])
    game.homeTeam = [onto[GameInfo["homeCode"]]]
    game.awayTeam = [onto[GameInfo["awayCode"]]]

    when = str(GameInfo["date"])

    year = int(when[:4])
    month = int(when[4:6])
    day = int(when[6:])

    if not (onto[year]):
        year = onto.Year(year)
    if not (onto[month]):
        month = onto.Month(month)
    if not (onto[day]):
        day = onto.Day(day)

    game.inYear = [year]
    game.inMonth = [month]
    game.inDay = [day]

    onto.save()

def create_player(onto, GameInfo, players, isaway, isbatter):
    for info in players:
        if isbatter == 1:
            player = onto.Hitter(info["name"] + info["pCode"])
            if isaway == 1:
                player.currentTeam = [onto[GameInfo["awayCode"]]]
            else:
                player.currentTeam = [onto[GameInfo["homeCode"]]]

            player.hasBirth = [int(info["birth"])]
            player.thisAVG = [float(info["seasonHra"])]

        else:
            player = onto.Pitcher(info["name"] + info["pCode"])
            if isaway == 1:
                player.currentTeam = [onto[GameInfo["homeCode"]]]
            else:
                player.currentTeam = [onto[GameInfo["awayCode"]]]

            player.hasBirth = [int(info["birth"])]
            player.thisERA = [float(info["seasonEra"])]

        onto.save()

def create_inn(onto, GameInfo, inn, btop):
    if(btop == 1):
        inning = onto.Inning(GameInfo["DateHomeAway"] + "_" + str(inn).zfill(2)+"초")
    else:
        inning = onto.Inning(GameInfo["DateHomeAway"] + "_" + str(inn).zfill(2)+"말")

    inning.hasInning = [inn]
    onto.save()

    return inning

def create_batterbox(onto, GameInfo, num_BatterBox, batter, pitcher, batorder, stay, inn_instance, btop):
    batterbox = onto.BatterBox(GameInfo["DateHomeAway"] + "_BatterBox_" + str(num_BatterBox).zfill(3))

    batter_pCode = batter["name"] + batter["pCode"]
    pitcher_pCode = pitcher["name"] + pitcher["pCode"]

    toHitter = onto[batter_pCode]
    fromPitcher = onto[pitcher_pCode]

    batterbox.toHitter = [toHitter]
    batterbox.fromPitcher = [fromPitcher]


    batterbox.inGame = [onto[GameInfo["DateHomeAway"]]]
    batterbox.inInning = [inn_instance]
    batterbox.hasOrder = [batorder]

    if(stay[0] != []):
        player = stay[0][0]
        player_pCode = player["name"] + player["pCode"]
        batterbox.stayIn1stBase = [onto[player_pCode]]

    if(stay[1] != []):
        player = stay[1][0]
        player_pCode = player["name"] + player["pCode"]
        batterbox.stayIn2ndBase = [onto[player_pCode]]

    if(stay[2] != []):
        player = stay[2][0]
        player_pCode = player["name"] + player["pCode"]
        batterbox.stayIn3rdBase = [onto[player_pCode]]

    onto.save()

    return batterbox, pitcher_pCode, batter_pCode

def create_pitchingbatting(onto, GameInfo, ball, batterbox, seq):
    if(ball == "strike"):
        ball = onto.Strike(GameInfo["DateHomeAway"] + "_Strike_" + str(seq).zfill(3))

    elif(ball == "ball"):
        ball = onto.Ball(GameInfo["DateHomeAway"] + "_Ball_" + str(seq).zfill(3))

    elif(ball == "foul"):
        ball = onto.Foul(GameInfo["DateHomeAway"] + "_Foul_" + str(seq).zfill(3))

    ball.inBatterbox = [batterbox]

    onto.save()

def create_out(onto, GameInfo, out, batterbox, seq):
    if(out == "strikeout"):
        out = onto.Strikeout(GameInfo["DateHomeAway"] + "_Strikeout_" + str(seq).zfill(3))

    elif(out == "outinbase"):
        out = onto.OutInBase(GameInfo["DateHomeAway"] + "_OutInBase_" + str(seq).zfill(3))

    elif(out == "fly"):
        out = onto.Fly(GameInfo["DateHomeAway"] + "_Fly_" + str(seq).zfill(3))

    else:
        out = onto.Out(GameInfo["DateHomeAway"] + "_Out_" + str(seq).zfill(3))

    batterbox.result.append(out)
    onto.save()


def create_hit(onto, GameInfo, hit, batterbox, seq):
    batter = batterbox.toHitter[0]

    if(hit == "fourball"):
        hit = onto.BaseOnBalls(GameInfo["DateHomeAway"] + "_BaseOnBalls_" + str(seq).zfill(3))
        batterbox.arrivedTo1stBase.append(batter)

    elif(hit == "hitbypitch"):
        hit = onto.HitByPitch(GameInfo["DateHomeAway"] + "_HitByPitch_" + str(seq).zfill(3))
        batterbox.arrivedTo1stBase.append(batter)

    elif(hit == "errorwalk"):
        hit = onto.BaseByError(GameInfo["DateHomeAway"] + "_BaseByError_" + str(seq).zfill(3))
        batterbox.arrivedTo1stBase.append(batter)

    elif(hit == "groundballwalk"):
        hit = onto.GetOnBase(GameInfo["DateHomeAway"] + "_GetOnBase_" + str(seq).zfill(3))
        batterbox.arrivedTo1stBase.append(batter)

    elif(hit == "singlehit"):
        hit = onto.SingleHit(GameInfo["DateHomeAway"] + "_SingleHit_" + str(seq).zfill(3))
        batterbox.arrivedTo1stBase.append(batter)

    elif(hit == "doublehit"):
        hit = onto.Double(GameInfo["DateHomeAway"] + "_Double_" + str(seq).zfill(3))
        batterbox.arrivedTo2ndBase.append(batter)

    elif(hit == "triplehit"):
        hit = onto.Triple(GameInfo["DateHomeAway"] + "_Triple_" + str(seq).zfill(3))
        batterbox.arrivedTo3rdBase.append(batter)

    elif(hit == "homerun"):
        hit = onto.HomeRun(GameInfo["DateHomeAway"] + "_HomeRun_" + str(seq).zfill(3))
        batterbox.passedToHomeBase.append(batter)

    batterbox.result.append(hit)

    onto.save()

def create_run(onto, GameInfo, run, dest, runner, batterbox, seq):
    runnerName_pCode = runner["name"] + runner["pCode"]
    runner = onto[runnerName_pCode]

    if(run == "steal"):
        run = onto.Steal(GameInfo["DateHomeAway"] + "_Steal_" + str(seq).zfill(3))

        if(dest == "1"):
            run.arrivedTo1stBase.append(runner)
        elif(dest == "2"):
            run.arrivedTo2ndBase.append(runner)
        elif(dest == "3"):
            run.arrivedTo3rdBase.append(runner)

        batterbox.stealed.append(run)

    elif(run == "homein"):
        batterbox.passedToHomeBase.append(runner)

    elif(run == "runBase"):
        if (dest == "1"):
            batterbox.arrivedTo1stBase.append(runner)
        elif (dest == "2"):
            batterbox.arrivedTo2ndBase.append(runner)
        elif (dest == "3"):
            batterbox.arrivedTo3rdBase.append(runner)

    onto.save()

def create_change(onto, GameInfo, player_in, player_out, seq):
    player_in = player_in["name"] + player_in["pCode"]
    player_out = player_out["name"] + player_out["pCode"]

    player_in = onto[player_in]
    player_out = onto[player_out]

    change = onto.Change(GameInfo["DateHomeAway"] + "_Change_" + str(seq).zfill(3))
    change.playIn = [player_in]
    change.playOut = [player_out]

    onto.save()

def search_pitcher(gameCode, p):
    g = rdflib.Graph()
    g.load('../_data/_owl/180515SKOB.owl')
    uri = "http://ailab.hanyang.ac.kr/ontology/baseball#"

    inGame = rdflib.URIRef(uri + "inGame")
    thisGame = rdflib.URIRef(uri + gameCode)

    fromPitcher = rdflib.URIRef(uri + "fromPitcher")
    pitcher = rdflib.URIRef(uri + p)
    thisERA = rdflib.URIRef(uri + "thisERA")

    result = rdflib.URIRef(uri + "result")

    query = "SELECT ?o where {?pitcher ?thisERA ?o}"
    r = g.query(query, initBindings={"pitcher": pitcher, "thisERA": thisERA})

    era = 0
    for row in r:
        era = row[0]

    query = "SELECT ?o where {?s ?fromPitcher ?pitcher . ?s ?inGame ?thisGame . ?s ?result ?o}"
    r = g.query(query, initBindings={"fromPitcher": fromPitcher, "pitcher": pitcher,
                                     "inGame": inGame, "thisGame": thisGame,
                                     "result": result})
    strikeout = 0
    baseonballs = 0

    for row in r:
        if ("Strikeout" in row[0]):
            strikeout = strikeout + 1
        if ("BaseOnBalls" in row[0]):
            baseonballs = baseonballs + 1

    annotation = [
        p + " 투수 오늘 경기 "+str(len(r))+"번째 타석에서 공을 던지고 있습니다.",
        p + " 투수 오늘 경기 "+str(strikeout)+"개의 삼진을 잡아내고 있습니다.",
        p + " 투수 오늘 경기 "+str(baseonballs)+"개의 포볼로 타자를 진루 시켰습니다.",
        p + " 투수 이번 시즌 "+str(era)+"의 평균 자책점을 기록하고 있습니다.",
        p + " 투수 과연 어떤 공을 던질까요?",
    ]

    return annotation

def search_batter(gameCode, b):
    g = rdflib.Graph()
    g.load('../_data/_owl/180515SKOB.owl')
    uri = "http://ailab.hanyang.ac.kr/ontology/baseball#"

    inGame = rdflib.URIRef(uri + "inGame")
    thisGame = rdflib.URIRef(uri + gameCode)

    toHitter = rdflib.URIRef(uri + "toHitter")
    batter = rdflib.URIRef(uri + b)
    thisAVG = rdflib.URIRef(uri + "thisAVG")

    result = rdflib.URIRef(uri + "result")
    stayIn1stBase = rdflib.URIRef(uri + "stayIn1stBase")

    query = "SELECT ?o where {?batter ?thisAVG ?o}"
    r = g.query(query, initBindings={"batter": batter, "thisAVG": thisAVG})

    avg = 0
    for row in r:
        avg = row[0]

    query = "SELECT ?o where {?s ?toHitter ?hitter . ?s ?inGame ?thisGame . ?s ?result ?o } order by ?s"
    r = g.query(query, initBindings={"toHitter": toHitter, "hitter": batter, "inGame": inGame, "thisGame": thisGame, "result": result})
    this_game_count = len(r)
    batter_history = []
    for row in r:
        batter_history.append(row[0].split("#")[1].split("_")[1])

    query = "SELECT ?o where {?s ?toHitter ?hitter . ?s ?inGame ?thisGame . ?s ?result ?o . ?s ?stayIn1stBase ?o1} order by ?s"
    r = g.query(query, initBindings={"toHitter": toHitter, "hitter": batter,
                                     "inGame": inGame, "thisGame": thisGame,
                                     "result": result, "stayIn1stBase": stayIn1stBase})
    batter_history_when1st = []
    for row in r:
        batter_history_when1st.append(row[0].split("#")[1].split("_")[1])

    annotation = [
        b + " 타자의 오늘 " + str(this_game_count) + "번째 타석입니다.",
        b + " 타자 오늘 " + str(this_game_count) + "번째 타석, " + ", ".join(_ for _ in batter_history) + "을 기록하고 있습니다.",
        b + " 타자 저번 타석, " + str(batter_history[-1]) + "을 기록하였습니다.",
        b + " 타자는 이번 시즌 " + str(avg) + "의 평균 타율을 기록하고 있습니다.",
        b + " 타자 오늘 경기 1루 주자가 있는 상황에서 " + ", ".join(_ for _ in batter_history_when1st) + "을 기록하고 있습니다.",
        "오늘 1루 주자가 있는 타석에서 " + b + " 타자 최근, " + str(batter_history_when1st[-1]) + "을 기록하였습니다.",
        b + " 타자 이번 타석 안타를 기록 할 수 있을까요?",
    ]

    return annotation

def search_pitcherbatter(gameCode, p, b):
    g = rdflib.Graph()
    g.load('../_data/_owl/180515SKOB.owl')
    uri = "http://ailab.hanyang.ac.kr/ontology/baseball#"

    inGame = rdflib.URIRef(uri + "inGame")
    thisGame = rdflib.URIRef(uri + gameCode)

    toHitter = rdflib.URIRef(uri + "toHitter")
    batter = rdflib.URIRef(uri + b)
    thisAVG = rdflib.URIRef(uri + "thisAVG")

    fromPitcher = rdflib.URIRef(uri + "fromPitcher")
    pitcher = rdflib.URIRef(uri + p)
    thisERA = rdflib.URIRef(uri + "thisERA")

    result = rdflib.URIRef(uri + "result")

    avg = 0
    era = 0
    query = "SELECT ?o where {?s ?p ?o}"
    r = g.query(query, initBindings={"s":batter, "p": thisAVG})
    for row in r:
        avg = row[0]

    r = g.query(query, initBindings={"s": pitcher, "p": thisERA})
    for row in r:
        era = row[0]

    query = "SELECT ?o where {?s ?inGame ?thisGame . ?s ?toHitter ?hitter . ?s ?fromPitcher ?pitcher . ?s ?result ?o} order by desc(?s)"
    r = g.query(query, initBindings={
        "inGame": inGame, "thisGame": thisGame,
        "toHitter": toHitter, "hitter": batter,
        "fromPitcher": fromPitcher, "pitcher": pitcher,
        "result": result
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

def search_runner(gameCode):

    return 1
search_pitcher("20180515OBSK", "후랭코프68240")
search_batter("20180515OBSK", "한동민62895")
search_pitcherbatter("20180515OBSK", "후랭코프68240", "한동민62895")
