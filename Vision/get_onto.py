import rdflib
import settings
from resource import Resource

class Annotation(object):
    def __init__(self, resource):
        self.resource = resource
        self.rdf = rdflib.Graph()
        self.rdf.load(settings.OWL_FILE_READONLY)
        self._set_instance()
        self._set_object_properties()
        self._set_data_properties()

    def reload(self):
        self.rdf.load(settings.OWL_FILE_READONLY)

    def _set_instance(self):
        self.BatterBox_uri = rdflib.URIRef(settings.OWL_URI + "BatterBox")

    def _set_object_properties(self):
        self.fromPitcher_uri = rdflib.URIRef(settings.OWL_URI + "fromPitcher")
        self.toBatter_uri = rdflib.URIRef(settings.OWL_URI + "toBatter")
        self.inBatterBox_uri = rdflib.URIRef(settings.OWL_URI + "inBatterBox")

        self.result_uri = rdflib.URIRef(settings.OWL_URI + "result")
        self.inGame_uri = rdflib.URIRef(settings.OWL_URI + "inGame")

        self.stayIn1B = rdflib.URIRef(settings.OWL_URI + "stayIn1B")
        self.stayIn2B = rdflib.URIRef(settings.OWL_URI + "stayIn2B")
        self.stayIn3B = rdflib.URIRef(settings.OWL_URI + "stayIn3B")

    def _set_data_properties(self):
        return 1

    def set_situation(self, scene_label, motion_label):
        return 1

    def about_batterbox(self, isaboutpitcher=None, isaboutbatter=None, isaboutrunner=None):
        self.resource.set_batterboxcode("531801_017")
        self.resource.set_gamecode("20180928_531801")
        bbox_uri = rdflib.URIRef(settings.OWL_URI + self.resource.get_batterboxcode())
        thisGame = rdflib.URIRef(settings.OWL_URI + self.resource.get_gamecode())

        annotation = []
        if(isaboutpitcher):
            query = "SELECT ?o WHERE {?batterbox ?fromPitcher ?o}"
            result = self.rdf.query(query, initBindings={"batterbox":bbox_uri, "fromPitcher":self.fromPitcher_uri})
            player = self._get_top_n_uri_inresult_to_list(result, 1)
            annotation.append(self.about_player(player[0][0]))

        if(isaboutbatter):
            query = "SELECT ?o WHERE {?batterbox ?toBatter ?o}"
            result = self.rdf.query(query, initBindings={"batterbox": bbox_uri, "toBatter": self.toBatter_uri})
            player = self._get_top_n_uri_inresult_to_list(result, 1)
            annotation.append(self.about_player(player[0][0]))

        if(isaboutrunner):
            query = "SELECT ?p ?o WHERE {?batterbox ?p ?o . FILTER (?p IN (?stayIn1B, ?stayIn2B, ?stayIn3B))}"
            result = self.rdf.query(query, initBindings={"batterbox": bbox_uri, "stayIn1B": self.stayIn1B, "stayIn2B": self.stayIn2B, "stayIn3B": self.stayIn3B})
            player = self._get_top_n_uri_inresult_to_list(result, 3)

        if(isaboutpitcher and isaboutbatter):
            query = "SELECT ?result_o WHERE {?batterbox ?toBatter ?batter . ?batterbox ?fromPitcher ?pitcher . ?other ?toBatter ?batter . ?other ?fromPitcher ?pitcher . ?other ?result_p ?result_o . ?batterbox ?inGame ?thisGame} order by ?other"
            result = self.rdf.query(query, initBindings={"batterbox": bbox_uri, "toBatter": self.toBatter_uri, "fromPitcher":self.fromPitcher_uri, "inBatterBox":self.inBatterBox_uri, "result_p":self.result_uri, "thisGame":thisGame, "inGame":self.inGame_uri})
            result_between_pitcherbatter_thisgame = self._get_top_n_uri_inresult_to_list(result)

        return annotation

    def about_player(self, player):
        thisGame = rdflib.URIRef(settings.OWL_URI + self.resource.get_gamecode())

        query = "SELECT ?p ?o WHERE {?player ?p ?o. ?p a owl:DatatypeProperty}"
        result = self.rdf.query(query, initBindings={"player": player})
        player_stat = self._get_top_n_uri_inresult_to_list(result)

        query = "SELECT ?result_o WHERE {?batterbox ?p ?player . FILTER (?p IN (?toBatter, ?fromPitcher)) . ?batterbox ?result_p ?result_o . ?batterbox ?inGame ?thisGame} order by desc(?batterbox)"
        result = self.rdf.query(query, initBindings={"player": player, "toBatter": self.toBatter_uri, "fromPitcher": self.fromPitcher_uri, "result_p": self.result_uri, "thisGame":thisGame, "inGame":self.inGame_uri})
        player_batterbox_history = self._get_top_n_uri_inresult_to_list(result)

        return ["",""]

    def about_game(self):
        return 1

    def _get_top_n_uri_inresult_to_list(self, result, n=None):
        if(n):
            return [row for row in result][:n]
        else:
            return [row for row in result]