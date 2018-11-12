# -*- coding: utf-8 -*-
import owlready2
import json
import operator
import requests
import time
from Annotation.EventData import *
from Annotation.Ontology_data import *
'''
relayText
        "inn":1,
        "btop":1,                                                => away attack
        
        "seqno":2,
        "pitchId":"180508_182923",
        "batstartorder":1, "batorder":1,
        "ilsun":0,                                              => nth batting
        
        "ballcount":1,
        "s":0, "b":1, "o":0,
        "homeBallFour":0, "awayBallFour":0,  
        
        
        "base3":0, "base1":0, "base2":0,
        
        "homeInningScore":0, "awayInningScore":0,
        "homeScore":0, "awayScore":0,
        "homeHit":0, "awayHit":0,
        "homeError":0, "awayError":0,
        
        "stuff":"SLID"
        
        "liveText":"1구 볼",
        "textStyle":1,
------------------------------------------------------------------------------------------------------------------------
ballData
        "inn":1,
        "pitchId":"180508_182923",
        
        "ballcount":1,
    
        "pitcherName":"소사",
        "batterName":"전준우",
        
        "crossPlateX":-0.171643,
        "topSz":3.68897,
        "crossPlateY":1.4167,
        "vy0":-124.287,
        "vz0":-1.36267,
        "vx0":3.1259,
        "z0":5.95536,
        "y0":50.0,
        "ax":-0.553696,
        "x0":-1.40222,
        "ay":26.1504,
        "az":-22.3011,
        "bottomSz":1.66,
        "stance":"R",
        
        "stuff":"슬라이더",
        "speed":"136"
'''

class RuleData():

    def __init__(self, gameName, Resources, onto):

        self.resources = Resources
        self.onto = onto

        fileName = "_data/"+gameName+"/"+gameName + ".txt"
        data_file = open(fileName, "rt", encoding="UTF8")
        data = json.load(data_file)
        data_file.close()

        ball_fileName = "_data/" + gameName + "/" + gameName + "_ball.txt"
        ball_data_file = open(ball_fileName, "rt", encoding="UTF8")
        self.ball_data = json.load(ball_data_file)
        ball_data_file.close()

        self.set_game_info(data["gameInfo"])
        self.set_TeamLineUp(home=data["homeTeamLineUp"], away=data["awayTeamLineUp"])

        self.relayTexts = self.set_relayTexts(data["relayTexts"])

    def set_game_info(self, game_info):
        '''"gameInfo"

        "aFullName":"롯데 자이언츠",
        "hPCode":"62698",
        "hCode":"LG",
        "hName":"LG",
        "cancelFlag":"N",
        "gdate":20180508,
        "aPCode":"68526",
        "round":4,
        "gtime":"18:30",
        "aName":"롯데",
        "gameFlag":"0",
        "hFullName":"LG 트윈스",
        "stadium":"잠실",
        "aCode":"LT",
        "optionFlag":1,
        "ptsFlag":"Y",
        "statusCode":"4"
        '''

        FhomeTeam = game_info["hFullName"]
        FawayTeam = game_info["aFullName"]

        homeTeam = game_info["hName"]
        awayTeam = game_info["aName"]

        homeCode = game_info["hCode"]
        awayCode = game_info["aCode"]

        date = game_info["gdate"]

        stadium = game_info["stadium"]

        self.GameInfo = {"FhomeTeam": FhomeTeam, "FawayTeam":FawayTeam, "homeTeam": homeTeam, "awayTeam": awayTeam, "stadium": stadium, "date" : date, "homeCode": homeCode, "awayCode": awayCode, "DateHomeAway" : str(date)+str(homeCode)+str(awayCode)}

        #input of ontology (game)

        self.resources.set_gameinfo(self.GameInfo)

        create_game(self.onto, self.GameInfo)
        return 1

    def set_TeamLineUp(self, home, away):

        homeTeamPitchers = home["pitcher"]
        homeTeamBatters = home["batter"]
        homeTeamBatters.sort(key=operator.itemgetter("batOrder"))

        create_player(self.onto, self.GameInfo, homeTeamPitchers, isaway=0, isbatter=0)
        create_player(self.onto, self.GameInfo, homeTeamBatters, isaway=0, isbatter=1)

        awayTeamPitchers = away["pitcher"]
        awayTeamBatters = away["batter"]
        awayTeamBatters.sort(key=operator.itemgetter("batOrder"))

        create_player(self.onto, self.GameInfo, awayTeamPitchers, isaway=1, isbatter=0)
        create_player(self.onto, self.GameInfo, awayTeamBatters, isaway=1, isbatter=1)

        self.LineUp = {"AwayPitchers": awayTeamPitchers, "AwayBatters": awayTeamBatters, "HomePitchers": homeTeamPitchers, "HomeBatters": homeTeamBatters}
        self.resources.set_LineUp(self.LineUp)

    def set_relayTexts(self, relayTexts):
        newlist = []
        newlist = newlist + relayTexts['1']
        newlist = newlist + relayTexts['2']
        newlist = newlist + relayTexts['3']
        newlist = newlist + relayTexts['4']
        newlist = newlist + relayTexts['5']
        newlist = newlist + relayTexts['6']
        newlist = newlist + relayTexts['7']
        newlist = newlist + relayTexts['8']
        newlist = newlist + relayTexts['9']
        newlist = newlist + relayTexts['currentBatterTexts']
        newlist = newlist + [relayTexts['currentOffensiveTeam']]
        newlist = newlist + [relayTexts['currentBatter']]

        newlist.sort(key=operator.itemgetter("seqno"))

        return newlist

    def get_time_delta_between_two_pichId(self, A, B):
        A_h = A[:2]
        A_m = A[2:4]
        A_s = A[4:]

        B_h = B[:2]
        B_m = B[2:4]
        B_s = B[4:]

        return 3600 * (int(B_h) - int(A_h)) + 60 * (int(B_m) - int(A_m)) + (int(B_s) - int(A_s))

    def add_time_delta_between_two_pichId(self, A, B):
        A_h = A[:2]
        A_m = A[2:4]
        A_s = A[4:]

        B_h = B[:2]
        B_m = B[2:4]
        B_s = B[4:]

        return 3600 * (int(B_h) + int(A_h)) + 60 * (int(B_m) + int(A_m)) + (int(B_s) + int(A_s))

    def secondTotime(self, sec):
        h = format(sec // 3600, '02')
        m = format((sec % 3600) // 60, '02')
        s = format(sec % 60, '02')
        sec = h + m + s
        return sec

    def set_Start(self, count_delta, fps, o_start):
        start = int(count_delta / fps)
        start = self.secondTotime(start)
        start = self.add_time_delta_between_two_pichId(o_start, start)
        start = self.secondTotime(start)
        self.start_pitchId = start
        no = 0

        self.PB = PitchingBatting(self.GameInfo, self.LineUp, self.onto, self.resources)
        self.C = Change(self.GameInfo, self.LineUp, self.onto, self.resources)
        self.R = Result(self.GameInfo, self.LineUp, self.onto, self.resources)

        for relayText in self.relayTexts:
            if (int(relayText["pitchId"].split("_")[-1]) > int(start)):
                no = relayText["seqno"]
                break
            pitchId = relayText["pitchId"]
            ball_data = self.find_ball_data_with_pitchId(pitchId)

            self.resources.set_gamescore(relayText["homeScore"], relayText["awayScore"])

            if (ball_data is None):
                if (relayText["ballcount"] == 0):  # 모든 교체(수비위치, 타석, 주자, 팀공격)
                    self.C.set(relayText)
                else:
                    self.R.set(relayText)

            else:  # pitching and batting
                self.PB.set(relayText, ball_data)
                pre_pitchId = pitchId

        self.relayTexts = self.relayTexts[no:]

    def find_ball_data_with_pitchId(self, pitchId):
        for i in self.ball_data:
            if(pitchId == i["pitchId"]):
                return i

        return None

    def get_Annotation(self):
        pre_pitchId = "000000_"+str(self.start_pitchId)
        #pre_pitchId = self.relayTexts[0]["pitchId"]

        for relayText in self.relayTexts:

            pitchId = relayText["pitchId"]
            ball_data = self.find_ball_data_with_pitchId(pitchId)

            self.resources.set_gamescore(relayText["homeScore"], relayText["awayScore"])

            if (ball_data is None):
                if(relayText["ballcount"] == 0): #모든 교체(수비위치, 타석, 주자, 팀공격)
                    annotation = self.C.set(relayText)
                else:
                    annotation = self.R.set(relayText)

            else:  # pitching and batting
                interval = self.get_time_delta_between_two_pichId(pre_pitchId.split("_")[-1], pitchId.split("_")[-1])
                #time.sleep(interval)

                annotation = self.PB.set(relayText, ball_data)

                pre_pitchId = pitchId

            next = input("next : ")
            print("from rule\t\t", annotation)
            self.resources.set_annotation(annotation)


