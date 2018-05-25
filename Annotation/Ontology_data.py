import owlready2

class OntologyData():
    def __init__(self):
        owlready2.onto_path.append("../_data/_owl/")
        self.onto = owlready2.get_ontology("database.owl")
        self.onto.load()


    def create_instance(self):
        print(self.onto.Strike)
        no = 3
        test_strike = self.onto.Strike("20180525KTSK_Strike" + str(no))
        self.onto.save()


a = OntologyData()
a.create_instance()
