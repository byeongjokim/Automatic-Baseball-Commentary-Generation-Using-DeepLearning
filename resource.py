class Resource():
    frame = []

    exit = False

    seq = 1
    prev_annotation_text = ""
    prev_annotation = ""
    annotation = "중계방송 준비중입니다."

    break_time = False

    strike = 0
    ball = 0
    out = 0

    def set_annotation(self, annotation):
        self.prev_annotation = self.annotation
        self.annotation = annotation

    def is_new_annotation(self):
        if not (self.prev_annotation == self.annotation):
            self.prev_annotation = self.annotation
            return 1
        else:
            return 0

    def is_new_annotation_video(self):
        if not (self.prev_annotation_text == self.annotation):
            self.prev_annotation_text = self.annotation
            return 1
        else:
            return 0

    def get_annotation(self):
        if(self.break_time):
            return "중계방송 준비중입니다."
        else:
            return self.annotation

    def set_frame(self, frame):
        self.frame = frame

    def get_frame(self):
        return self.frame

    def set_exit(self, exit):
        self.exit = exit

    def add_seq(self):
        self.seq = self.seq + 1

    def get_seq(self):
        return self.seq

    def set_gameinfo(self, game_info):
        self.stadium = game_info["stadium"]
        self.date = game_info["date"]
        self.DateHomeAway = game_info["DateHomeAway"]

        self.home_name = game_info["homeTeam"]
        self.away_name = game_info["awayTeam"]

        self.home_Fname = game_info["FhomeTeam"]
        self.away_Fname = game_info["FawayTeam"]

    def get_gameinfo(self):
        return [self.stadium, self.date, [self.home_name, self.away_name], [self.home_Fname, self.away_Fname]]

    def set_gamescore(self, homescore, awayscore):
        self.homescore = homescore
        self.awayscore = awayscore

    def get_gamescore(self):
        return [self.homescore, self.awayscore]

    def get_date(self):
        return self.date

    def get_stadium(self):
        return self.stadium

    def get_gamecode(self):
        return self.DateHomeAway

    def clear_strike_ball_out(self):
        self.strike = 0
        self.ball = 0
        self.out = 0

    def set_strike_ball_out(self, strike=False, ball=False, foul=False, out=False):
        if(strike):
            self.strike = self.strike + 1
        elif(ball):
            self.ball = self.ball + 1
        elif(foul):
            if(self.strike < 2):
                self.strike = self.strike + 1
        elif(out):
            self.out = self.out + 1

    def get_strike_ball_out(self):
        return [self.strike, self.ball, self.out]

    def set_batterbox(self, batterbox, pitcher, batter):
        self.batterbox = batterbox

        self.pitcher = pitcher
        self.batter = batter

    def get_batterbox(self):
        return self.batterbox

    def get_pitcher(self):
        return self.pitcher

    def set_batter(self, batter):
        self.batter = batter

    def get_batter(self):
        return self.batter

    def set_inn(self, inn, btop):
        self.inn = inn
        self.btop = btop

    def get_inn(self):
        return self.inn

    def get_btop(self):
        return self.btop

    def set_LineUp(self, LineUp):

        sett = {"catcher":None, "1st":None, "2nd":None, "3rd":None, "ss":None, "ROF":None, "LOF":None, "COF":None}

        hometeam = LineUp["HomePitchers"] + LineUp["HomeBatters"]
        awayteam = LineUp["AwayPitchers"] + LineUp["AwayBatters"]

        for player in LineUp["HomeBatters"]:
            if(player["posName"] == "1루수"):
                sett["1st"] = player
            elif(player["posName"] == "2루수"):
                sett["2nd"] = player
            elif (player["posName"] == "3루수"):
                sett["3rd"] = player
            elif (player["posName"] == "유격수"):
                sett["ss"] = player
            elif (player["posName"] == "포수"):
                sett["catcher"] = player
            elif (player["posName"] == "좌익수"):
                sett["LOF"] = player
            elif (player["posName"] == "우익수"):
                sett["ROF"] = player
            elif (player["posName"] == "중견수"):
                sett["COF"] = player

        self.homeTeam = sett

        sett = {"catcher":None, "1st":None, "2nd":None, "3rd":None, "ss":None, "ROF":None, "LOF":None, "COF":None}

        for player in LineUp["AwayBatters"]:
            if(player["posName"] == "1루수"):
                sett["1st"] = player
            elif(player["posName"] == "2루수"):
                sett["2nd"] = player
            elif (player["posName"] == "3루수"):
                sett["3rd"] = player
            elif (player["posName"] == "유격수"):
                sett["ss"] = player
            elif (player["posName"] == "포수"):
                sett["catcher"] = player
            elif (player["posName"] == "좌익수"):
                sett["LOF"] = player
            elif (player["posName"] == "우익수"):
                sett["ROF"] = player
            elif (player["posName"] == "중견수"):
                sett["COF"] = player

        self.awayTeam = sett

        self.LineUp = LineUp

    def get_LineUp(self, homeTeam = 1):
        if homeTeam == 1:
            return self.homeTeam
        else:
            return self.awayTeam

    def change_LineUp(self, player_in, player_out):
        pos = player_in["posName"]

        if (pos == "1루수"):
            if (self.homeTeam["1st"] == player_out):
                self.homeTeam["1st"] = player_in
            if (self.awayTeam["1st"] == player_out):
                self.awayTeam["1st"] = player_in
        elif (pos == "2루수"):
            if (self.homeTeam["2nd"] == player_out):
                self.homeTeam["2nd"] = player_in
            if (self.awayTeam["2nd"] == player_out):
                self.awayTeam["2nd"] = player_in
        elif (pos == "3루수"):
            if (self.homeTeam["3rd"] == player_out):
                self.homeTeam["3rd"] = player_in
            if (self.awayTeam["3rd"] == player_out):
                self.awayTeam["3rd"] = player_in
        elif (pos == "dbrurtn"):
            if (self.homeTeam["ss"] == player_out):
                self.homeTeam["ss"] = player_in
            if (self.awayTeam["ss"] == player_out):
                self.awayTeam["ss"] = player_in
        elif (pos == "포수"):
            if (self.homeTeam["catcher"] == player_out):
                self.homeTeam["catcher"] = player_in
            if (self.awayTeam["catcher"] == player_out):
                self.awayTeam["catcher"] = player_in
        elif (pos == "좌익수"):
            if (self.homeTeam["LOF"] == player_out):
                self.homeTeam["LOF"] = player_in
            if (self.awayTeam["LOF"] == player_out):
                self.awayTeam["LOF"] = player_in
        elif (pos == "우익수"):
            if (self.homeTeam["ROF"] == player_out):
                self.homeTeam["ROF"] = player_in
            if (self.awayTeam["ROF"] == player_out):
                self.awayTeam["ROF"] = player_in
        elif (pos == "중견수"):
            if (self.homeTeam["COF"] == player_out):
                self.homeTeam["COF"] = player_in
            if (self.awayTeam["COF"] == player_out):
                self.awayTeam["COF"] = player_in

    def get_player_with_position(self, position):
        btop = self.get_btop()
        if btop == 0:
            lineup = self.get_LineUp(0)
        else:
            lineup = self.get_LineUp(1)

        return lineup[position]["name"]