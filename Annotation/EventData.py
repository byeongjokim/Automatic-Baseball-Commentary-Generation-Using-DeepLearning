import json

class PitchingBattingData():
    def __init__(self, game):
        print("PitchingBattingData init")
        self.balldata = ''

    '''
    {
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
    }
    
    {
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
    }
    '''
    def set_json(self, js):
        self.js = js
        self.balldata =

    def set_event(self):
        '''
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
        '''

        self.inn = self.js["inn"]
        self.btop = self.js["btop"]

        self.pitchId = self.js["pitchId"]
        self.batorder = self.js["batorder"]
        self.ilsun = self.js["ilsun"]




