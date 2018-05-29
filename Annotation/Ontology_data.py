import owlready2

class OntologyData():
    def __init__(self):
        owlready2.onto_path.append("../_data/_owl/")
        self.onto = owlready2.get_ontology("database.owl")
        self.onto.load()

    def create_player(self, onto, info, isbatter):
        if isbatter == 1:
            player = self.onto.Hitter(info["name"] + info["pCode"])
            player.asdf = ["asd"]
            player.hasBirth = ["19931031"]

        else:
            player = onto.Pitcher(info["name"] + info["pCode"])

        self.onto.save()

    def create_batterbox(self, onto, GameInfo, num_BatterBox, batter, pitcher):
        batterbox = onto.BatterBox(GameInfo["DateHomeAway"] + "_BatterBox_" + str(num_BatterBox))

        batter_pCode = batter["name"] + batter["pCode"]
        pitcher_pCode = pitcher["name"] + pitcher["pCode"]

        if (onto[batter_pCode]):
            batterbox.toHitter = [onto[batter_pCode]]
        else:
            batterbox.toHitter = [onto.Hitter(batter_pCode)]

        if (onto[pitcher_pCode]):
            batterbox.fromPitcher = [onto[pitcher_pCode]]
        else:
            batterbox.fromPitcher = [onto.Pitcher(pitcher_pCode)]

        onto.save()

    #def update_player(self, info):



a = OntologyData()
test = {"bb":0,"ab":1,"posName":"우익수","seqno":2,"cout":"true","batOrder":9,"seasonHra":0.318,"birth":"19930830","weight":"73.0","run":0,"hr":0,"psHra":0,"vsHra":"0.429","backnum":"51","hit":0,"hitType":"우투좌타","pos":9,"pCode":"66209","hbp":0,"name":"가나다","rbi":0,"todayHra":0,"so":1,"height":"178.0"}
a.create_player(1, test, 1)
