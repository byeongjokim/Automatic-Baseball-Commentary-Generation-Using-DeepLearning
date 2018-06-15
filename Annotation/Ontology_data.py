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

def test():
    g = rdflib.Graph()
    g.load('../_data/_owl/180515SKOB.owl')

    uri = "http://ailab.hanyang.ac.kr/ontology/baseball#"

    batterbox = rdflib.URIRef(uri + "BatterBox")
    result = rdflib.URIRef(uri + "result")
    toHitter = rdflib.URIRef(uri + "toHitter")
    hitter = rdflib.URIRef(uri + "정의윤75151")
    fly = rdflib.URIRef(uri + "Fly")

    q = "select ?s ?o where {?s ?result ?o . ?s ?toHitter ?hitter . filter contains(str(?o), 'Fly')}"
    r = g.query(q, initBindings={"result":result, "toHitter":toHitter, "hitter":hitter, "type":fly})


    #q = "select ?s ?o1 where {?s ?toHitter ?hitter . ?s ?result ?o1}"
    #r = g.query(q, initBindings={"result": result, "toHitter": toHitter, "hitter": hitter})

    for row in r:
        print(row)

def search_batterbox(g, query):
    uri = "http://ailab.hanyang.ac.kr/ontology/baseball#"

    inGame = rdflib.URIRef(uri + "inGame")
    inInning = rdflib.URIRef(uri + "inInning")

    batterbox = rdflib.URIRef(uri + "BatterBox")
    fromPitcher = rdflib.URIRef(uri + "fromPitcher")
    toHitter = rdflib.URIRef(uri + "toHitter")
    result = rdflib.URIRef(uri + "result")

    return 1

def search_stat_of_player(g):
    uri = "http://ailab.hanyang.ac.kr/ontology/baseball#"

    b = "오재원77248"
    p = "켈리65856"
    gamecode = "20180515OBSK"

    thisAVG = rdflib.URIRef(uri + "thisAVG")
    toHitter = rdflib.URIRef(uri + "toHitter")
    fromPitcher = rdflib.URIRef(uri + "fromPitcher")

    result = rdflib.URIRef(uri + "result")
    strikeout = rdflib.URIRef(uri + "Strikeout")

    inGame = rdflib.URIRef(uri + "inGame")
    thisGame = rdflib.URIRef(uri + gamecode)

    batter = rdflib.URIRef(uri + b)
    pitcher = rdflib.URIRef(uri + p)

    query = "SELECT ?o where {?s ?toHitter ?batter . ?s ?inGame ?thisGame . ?s ?result ?o}"
    r = g.query(query, initBindings={"toHitter":toHitter, "batter":batter,
                                     "inGame":inGame,"thisGame":thisGame,
                                     "result":result})
    this_game_count = len(r)
    batter_history = []
    for row in r:
        batter_history.append(row[0].split("#")[1].split("_")[1])

    print(b + " 타자 오늘 " + str(this_game_count) +" 번째 타석, " + ", ".join(_ for _ in batter_history) + "을 기록하고 있습니다.")

g = rdflib.Graph()
g.load('../_data/_owl/180515SKOB.owl')

search_stat_of_player(g)
#search_batterbox(g, "정의윤75151")







