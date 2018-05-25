import random

#######################################################################################################BatterBox
def BatterBox():
    annotation = [
        "지난 경기에 A 투수는 B 타자에게 3개의 안타를 허용했습니다.",
        "지난 경기에 B 타자는 A 투수에게 2번의 삼진아웃과, 1번의 안타를 쳤네요",
        "투수와 타자의 신경전 속에 투수는 어떤 공을 선택 할까요?",
    ]
    return random.choice(annotation)

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