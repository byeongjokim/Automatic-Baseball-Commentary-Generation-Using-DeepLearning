class Person():
    frame_no = 0

    def __init__(self, Annotation, Resources):
        print("init_person")
        self.Resources = Resources

        self.Annotation = Annotation
        self.awayTeamLineUp = self.Annotation.naverData.awayTeamLineUp
        self.homeTeamLineUp = self.Annotation.naverData.homeTeamLineUp

