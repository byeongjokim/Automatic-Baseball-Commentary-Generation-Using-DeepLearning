import random

#######################################################################################################ballCount
def noStrikeBallCount(ball):
    if(ball == 0):
        b = "노"
    elif(ball == 1):
        b = "원"
    elif(ball == 2):
        b = "투"
    elif(ball == 3):
        b = "쓰리"

    annotation = [
        "카운트 " + b + " 볼.",
        "카운트 " + b + " 볼입니다.",
        "볼카운트 " + b + " 볼 노 스트라이크."
    ]
    return random.choice(annotation)

def noBallBallCount(strike):
    if(strike == 0):
        s = "노"
    elif(strike == 1):
        s = "원"
    elif(strike == 2):
        s = "투"

    annotation = [
        "카운트 노 볼 " + s + " 스트라이크.",
        "스트라이크 " + s + ".",
        "볼카운트는 노 볼 " + s + " 스트라이크.",
    ]
    return random.choice(annotation)

def fullCount():
    annotation = [
        "카운트 풀카운트.",
        "풀카운트 승부입니다."
    ]
    return random.choice(annotation)

def BallCount(strike, ball):
    if(ball == 0):
        return noBallBallCount(strike)
    elif(ball == 1):
        b = "원"
    elif(ball == 2):
        b = "투"
    elif(ball == 3):
        b = "쓰리"

    if(strike == 0):
        return noStrikeBallCount(ball)
    elif(strike == 1):
        s = "원"
    elif(strike == 2):
        s = "투"

    if(ball == 3 and strike == 2):
        return fullCount()

    annotation = [
        "카운트 " + b + " 볼 " + s + " 스트라이크.",
        "볼카운트는 " + b + " 볼, " + s + " 스트라이크."
    ]
    return random.choice(annotation)

#######################################################################################################strike
def difficultCourseStrike(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구는 스트라이크! 어려운 코스에 공이 꽂혔습니다.",
        "스트라이크! 구석에 꽂히는 " + stuff + "에 타자가 제대로 속았습니다.",
        ballCount + "구는 스트라이크입니다. 타자가 속았어요.",
        ballCount + "구 스트라이크! 이런 코스는 타자가 치기 어렵죠."
    ]
    return random.choice(annotation)

def easyCourseStrike(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구는 스트라이크! 공이 좀 몰렸는데요, 타자 입장에선 아쉽겠습니다.",
        "스트라이크! 한복판에 꽂히는 " + stuff + "에 타자가 제대로 반응하지 못 했네요.",
        ballCount + "구는 스트라이크입니다. 이런 공은 쳤어야 할 텐데요.",

    ]
    return random.choice(annotation)

def insideStrike(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구 몸쪽 스트라이크! 바싹 붙여봤습니다.",
        "몸쪽에 꽂히는 스트라이크! " + ballCount + "구는 스트라이크입니다."
    ]
    return random.choice(annotation)

def outsideStrike(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구 바깥쪽 스트라이크! 스트라이크 존에 바싹 붙여봤습니다.",
        "바깥쪽에 걸치는 스트라이크! " + ballCount + "구는 스트라이크입니다."
    ]
    return random.choice(annotation)

def upsideStrike(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구는 높은 공, 스트라이크! 위험한 공이었지만 스트라이크 들어갔습니다.",
        "높은 코스 스트라이크! " + ballCount + "구는 스트라이크네요."
    ]
    return random.choice(annotation)

def downsideStrike(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구 낮은 코스에 스트라이크! 낮았지만 공이 들어왔습니다.",
        "낮은 코스에 스트라이크에요. " + ballCount + "구는 스트라이크입니다."
    ]
    return random.choice(annotation)

def normalStrike(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구 스트라이크! " + Speed + "이(가) 찍혔습니다.",
        "투수가  " + stuff + "(으)로 스트라이크 잡아냈습니다."
    ]
    return random.choice(annotation)

#######################################################################################################ball
def difficultCourseBall(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구는 볼입니다. 조금 멀리 벗어났군요.",
        "외곽으로 벗어난 볼. " + stuff + "에 타자가 속지 않았습니다.",
        ballCount + "구 볼이군요.  이런 코스는 타자가 치기 어렵죠."
    ]
    return random.choice(annotation)

def easyCourseBall(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구는 볼입니다만, 이런 공에 볼 판정이 나오는군요."
    ]
    return random.choice(annotation)

def insideBall(ballCount, Speed, stuff):
    annotation = [
        "제 " + ballCount + "구, 몸쪽에 바짝 붙여봤습니다만, 심판은 콜을 하지 않았습니다.",
        "몸쪽에! 아 " + ballCount + "구는 볼이네요."
    ]
    return random.choice(annotation)

def outsideBall(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구는 볼입니다. 조금 빠진 것 같네요.",
        "바깥쪽으로 빠졌나요. " + ballCount + "구는 볼입니다."
    ]
    return random.choice(annotation)

def upsideBall(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구는 낮았군요. 볼입니다.",
        "높은 코스로 볼 들어 옵니다. " + ballCount + "구 볼."
    ]
    return random.choice(annotation)

def downsideBall(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구 낮은 코스에 볼이었습니다.",
        "낮은 코스로 들어오는 볼. " + ballCount + "구는 볼입니다."
    ]
    return random.choice(annotation)

def normalBall(ballCount, Speed, stuff):
    annotation = [
        "제 " + ballCount + "구 볼.",
        "투수가  " + stuff + "을(를) 던져봤지만, 타자 속지 않습니다.",
        ballCount + "구 볼. 타자가 속지 않았어요."
    ]
    return random.choice(annotation)

#######################################################################################################swing
def difficultCourseSwing(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구는 스윙 스트라이크! 크게 휘둘러 봤지만 맞지 않았습니다.",
        "스트라이크! 구석에 꽂히는 " + stuff + "에 타자가 제대로 속았습니다.",
        ballCount + "구, 휘둘러 봤지만 맞지 않습니다.",
        ballCount + "구 스윙 스트라이크! 이런 코스는 타자가 치기 어렵죠."
    ]
    return random.choice(annotation)

def easyCourseSwing(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구는 스윙 스트라이크! 공이 좀 몰렸는데요, 이게 맞질 않았습니다.",
        "스트라이크! 한복판에 꽂히는 " + stuff + "에 타자가 타이밍을 뺏긴 것 같습니다.",
        ballCount + "구에 헛스윙! 이런 공은 쳐줬어야 하는데요.",
    ]
    return random.choice(annotation)

def insideSwing(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구 몸쪽 공에 배트 돌아갑니다!",
        "몸쪽 공에 헛스윙! 배트가 헛돌았습니다."
    ]
    return random.choice(annotation)

def outsideSwing(ballCount, Speed, stuff):
    annotation = [
        "제 " + ballCount + "구, 바깥쪽 공에 스윙! 배트가 닿지 않았습니다.",
        "바깥쪽에 걸치는 공에 헛스윙 하고 맙니다."
    ]
    return random.choice(annotation)

def upsideSwing(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구는 높게! 배트가 닿지 않았습니다. 스트라이크.",
        "높은 코스에 배트 나갑니다. 타자로서는 아쉬운 상황입니다."
    ]
    return random.choice(annotation)

def downsideSwing(ballCount, Speed, stuff):
    annotation = [
        "제 " + ballCount+ "구 낮은 공에 배트 휘둘렀습니다.",
    ]
    return random.choice(annotation)

def normalSwing(ballCount, Speed, stuff):
    annotation = [
        ballCount + "구, 헛스윙! 구속 " + Speed + "이(가) 찍혔습니다.",
        "투수가  " + stuff + "(으)로 타자의 방망이를 유인해냈습니다. 스윙 스트라이크."
    ]
    return random.choice(annotation)

#######################################################################################################foul

def cut(ballCount, Speed, stuff):
    annotation = [
        "제 " + ballCount + "구. 다시 한 번 승부를 이어갑니다.",
        "제 " + ballCount + "구는 파울입니다. 끈질긴 승부를 이어가고 있습니다.",
        "파울! 벌써 " + ballCount + "구째 승부 중입니다."
    ]
    return random.choice(annotation)

def normalFoul(ballCount, Speed, stuff):
    annotation = [
        "제 " + ballCount + "구. 파울라인 바깥으로 나가면서 파울이 됩니다.",
        "제 " + ballCount + "구는 파울입니다.",
        "파울. 걷어냈습니다."
    ]
    return random.choice(annotation)

def buntFoul(ballCount, Speed, stuff):
    annotation = [
        "제 " + ballCount + "구. 번트를 댔는데 파울이 되고 맙니다.",
        ballCount + "구 번트! 아 아쉽게도 파울라인을 넘어갑니다.",
        "번트를 댔습니다만 파울이 됐습니다."
    ]
    return random.choice(annotation)

#######################################################################################################hit
def hit(ballCount, Speed, stuff):
    annotation = [
        "제 " + ballCount + "구. 쳤습니다!",
        ballCount + "구는 받아쳤습니다.",
        "쳤습니다!"
    ]
    return random.choice(annotation)

#######################################################################################################batter
def batter_Start(batter):
    batOrder = str(batter["batOrder"])
    Name = str(batter["name"])

    annotation = [
        batOrder + "번타자 " + Name + " 타석에 들어와 있습니다.",
        "타석에 " + batOrder + "번타자 " + Name + " 들어와 있습니다.",
        "자 다음 타자는 " + Name + "입니다. 오늘 " + batOrder + "번에 배치돼 있습니다."
    ]
    return random.choice(annotation)

'''
def batter_sesan_Stat(batter):
    annotation = [
        "이번 시즌 " + css.getHra() + "의 타율과 함께 " + css.getHr() + "홈런, " + css.getRbi() + "타점 기록하고 있습니다.",
        "올 시즌 " + css.getHr() + "홈런, " + css.getRbi() + "타점 기록하면서 " + css.getHra() + " 타율 기록 중입니다."
    ]
    return  random.choice(annotation)
'''

#######################################################################################################inn
def inn_start(GameInfo, inn, btop, score):
    if(btop == 1):
        inn = inn + 1
        teamName = GameInfo["awayTeam"]
        flag = "초"
        s = score["awayScore"]

    else:
        teamName = GameInfo["homeTeam"]
        flag = "말"
        s = score["homeScore"]

    annotation = [
        str(inn) + "회" + flag + " " + teamName + "의 공격이 시작 됩니다.",
        str(inn) + "회" + flag + " " + teamName + "의 공격이 "+ str(s) +"점을 가진채 시작 됩니다.",
    ]

    return random.choice(annotation)

def inn_end(GameInfo, inn, btop, score, innscore):
    if(btop == 1):
        teamName = GameInfo["homeTeam"]
        flag = "말"
        s = score["homeScore"]
        ins = innscore["homeInningScore"]

        inn = inn - 1

    else:
        teamName = GameInfo["awayTeam"]
        flag = "초"
        s = score["awayScore"]
        ins = innscore["awayInningScore"]

    ann1 = ""
    if inn != 0:
        annotation = [
            str(inn) + "회" + flag + " " + teamName + "의 공격이 마무리 됩니다.",
            str(inn) + "회" + flag + " " + teamName + "의 공격이 "+ str(s) +"점으로 마무리 됩니다.",
            str(inn) + "회" + flag + " " + teamName + " 이번 이닝 " + str(ins) + "점을 추가 획득 하였습니다.",
        ]
        ann1 = random.choice(annotation)


    return ann1 + "\n" + inn_start(GameInfo, inn, btop, score)

#######################################################################################################out
def strikeOut(batter):
    name = str(batter["name"])

    annotation = [
        "스트라이크 아웃! " + name + "선수는 삼진으로 물러납니다.",
        "삼진! 결국 " + name + "선수를 삼진으로 잡아내는군요.",
        "삼진~! " + name + "선수를 삼진 처리하면서 아웃카운트 하나를 따내는군요."
    ]
    return random.choice(annotation)

def lineDriveOut(batter, pos):
    name = str(batter["name"])
    annotation = [
        "아~ 잘 맞았지만 " + pos + "의 글러브에 빨려 들어갑니다. 라인드라이브 아웃으로 " + name + "선수는 들어갑니다."
    ]
    return random.choice(annotation)

def flyOut(batter, pos):
    name = str(batter["name"])
    annotation = [
        "뜬공~ " + pos + "가 잡아냅니다. 아웃!"
    ]
    return random.choice(annotation)

def groundBallOut(batter, pos):
    name = str(batter["name"])
    annotation = [
        pos + " 쪽으로 흐르는 볼. " + pos + "가 잡아서 1루에 던집니다.\n" + name + "선수는 땅볼로 물러납니다."
    ]
    return random.choice(annotation)

def doublePlayedOut(batter, pos):
    name = str(batter["name"])
    annotation = [
        pos + "부터 이어지는 병살타! " + name + "선수는 병살타를 치고 덕아웃으로 들어갑니다."
    ]
    return random.choice(annotation)

def strikeNotOut(batter, pos):
    name = str(batter["name"])
    annotation = [
        "낫 아웃!"
    ]
    return random.choice(annotation)

def sacrificeBunt(batter, pos):
    name = str(batter["name"])
    annotation = [
        pos + "가 잡아서 1루에 던집니다. " + name + "선수는 희생번트로 물러납니다."
    ]
    return random.choice(annotation)

def tagNforceOut(name):
    annotation = [
        name + "선수 아웃 됩니다."
    ]
    return random.choice(annotation)
#######################################################################################################안타
def fourBall(name):
    annotation = [
        name + "선수는 볼넷으로 걸어나갑니다.",
        name + "선수, 끈질긴 승부 끝에 볼넷을 얻어냅니다."
    ]
    return random.choice(annotation)

def hitByPitch(name):
    annotation = [
        "아, 몸에 맞았군요. " + name + "선수 몸에 맞는 볼로 출루합니다.",
        name + "선수 몸에 맞았어요. 몸에 맞는 볼로 걸어 나가네요."
    ]
    return random.choice(annotation)

def intentionalBaseOnBalls(name):
    annotatino = [
        name + "선수는 고의사구로 걸어나갑니다.",
        "배터리가 " + name + "선수를 거르는군요. " + name + "선수, 고의사구로 출루합니다."
    ]
    return random.choice(annotatino)

def inFieldHit(name, pos):
    annotation = [
        pos + "가 던져봤지만 세이프입니다. " + name + "선수 빠른 발로 내야안타 만들어냅니다."
    ]
    return random.choice(annotation)

def outFieldHit(name, pos):
    if (name == "안익훈"):
        name = "이형종"
    if("중간" in pos):
        annotation = [
            pos + "에 떨어지는 안타 만들어내면서 " + name + "선수 1루에 안착합니다."
        ]
    else:
        annotation = [
            pos + " 앞에 떨어지는 안타! " + name + "선수 안타 만들어냅니다."
        ]
    return random.choice(annotation)

def outFieldDoubleHit(name, pos):
    annotation = [
        pos + " 뒤로 넘어가는군요! " + name + "선수 2루타!"
    ]
    return random.choice(annotation)

def outFieldTripleHit(name, pos):
    annotation = [
        pos + "가 잡지 못 했거든요! 그 사이에 " + name + "선수는 3루까지!"
    ]
    return random.choice(annotation)

def groundballWalk(name, pos):
    annotation = [
        pos + " 쪽으로 흘러갑니다. 다른 주자 아웃되는 사이에 " + name + "선수는 1루에 들어섭니다."
    ]
    return random.choice(annotation)

def errorWalk(name, pos):
    annotation = [
        pos + " 의 실수로 " + name + "선수는 1루에 들어섭니다."
    ]
    return random.choice(annotation)

def HomeRun(name, pos):
    annotation = [
        pos + " 를 넘어갑니다!!! " + name + "선수 홈런~~!"
    ]
    return random.choice(annotation)

#######################################################################################################run
def runHome(origin, name, aScore, hScore):
    annotation = [
        origin + "에 있던 " + name + "은(는) 홈으로 들어옵니다!",
        origin + "주자 " + name + "은(는) 홈에 안착합니다!",
        name + "은(는) " + origin + "에서 홈까지 진루했습니다."
    ]

    annotation2 = [
        "스코어 " + aScore + ":" + hScore + "!",
        "스코어는 " + aScore + ":" + hScore + "이 됩니다!"
    ]

    return random.choice(annotation) + " " + random.choice(annotation2)

def runBase(origin, destination, name):

    annotation = [
        origin + "에 있던 " + name + "은(는) " + destination + "까지 들어갑니다.",
        origin + "주자 " + name + "은(는) " + destination + "에 안착합니다.",
        name + "은(는) " + origin + "에서 " + destination + "까지 진루했습니다."
    ]
    return random.choice(annotation)

def thiefBase(origin, destination, name):

    annotation = [
        origin + "에 있던 " + name + "은(는) 도루로 " + destination + "까지 들어갑니다.",
        origin + "주자 " + name + "은(는) 도루로 " + destination + "에 안착합니다.",
        name + "은(는) " + origin + "에서 도루로 " + destination + "까지 진루했습니다."
    ]
    return random.choice(annotation)

def runError(origin, destination, name):
    annotation = [
        "아~ 수비 실책으로 " + origin + "에 있던 " + name + ", " + destination + "까지 들어갑니다.",
        "수비의 실책 때문에 " + origin + "주자 " + destination + "에 안착합니다.",
        name + "은(는) " + origin + "에서 " + destination + "까지 진루했습니다. 수비의 실책이 아쉬운 부분입니다."
    ]
    return random.choice(annotation)

