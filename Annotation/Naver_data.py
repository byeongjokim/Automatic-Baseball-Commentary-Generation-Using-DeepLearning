# -*- coding: utf-8 -*-
import json
import operator
import requests
import time
from Annotation.EventData import *
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

class NaverData():

    def __init__(self, gameName):

        fileName = "_data/"+gameName+"/"+gameName + ".txt"
        data_file = open(fileName, "rt", encoding="UTF8")
        data = json.load(data_file)
        data_file.close()

        ball_fileName = "_data/" + gameName + "/" + gameName + "_ball.txt"
        ball_data_file = open(ball_fileName, "rt", encoding="UTF8")
        self.ball_data = json.load(ball_data_file)
        ball_data_file.close()

        self.set_game_info(data["gameInfo"])

        self.awayTeamPitchers, self.awayTeamBatters = self.set_TeamLineUp(data["awayTeamLineUp"])
        self.homeTeamPitchers, self.homeTeamBatters = self.set_TeamLineUp(data["homeTeamLineUp"])

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

        print(stadium)
        #input of ontology (game)

        return 1

    def set_TeamLineUp(self, TeamLineUp):
        pitchers = TeamLineUp["pitcher"]
        batters = TeamLineUp["batter"]

        batters.sort(key=operator.itemgetter("batOrder"))

        return pitchers, batters


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

    def find_ball_data_with_pitchId(self, pitchId):
        for i in self.ball_data:
            if(pitchId == i["pitchId"]):
                return i

        return None

    def get_Annotation(self):
        a = PitchingBatting()
        for relayText in self.relayTexts:
            pitchId = relayText["pitchId"]
            ball_data = self.find_ball_data_with_pitchId(pitchId)

            if (ball_data is None):
                print("later")
                # 아웃
                # 교체
                # n번타자


            else:  # pitching and batting
                a.set(0, relayText, ball_data)

            print(relayText["liveText"])


