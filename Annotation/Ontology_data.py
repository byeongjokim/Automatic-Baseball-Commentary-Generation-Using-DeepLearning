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
            ####here
            if("스크럭스" in info["name"]):
                player.thisAVG = [0.256]
            if ("이원재" in info["name"]):
                player.thisAVG = [0.337]
            if ("정범모" in info["name"]):
                player.thisAVG = [0.166]

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
        inning = onto.Inning(GameInfo["DateHomeAway"] + "_" + str(inn) + "회초")
    else:
        inning = onto.Inning(GameInfo["DateHomeAway"] + "_" + str(inn) + "회말")

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
