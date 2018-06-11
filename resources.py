import numpy
class Resources():
    frame = []

    exit = False

    seq = 1


    def set_frame(self, frame):
        self.frame = frame

    def set_exit(self, exit):
        self.exit = exit

    def add_seq(self):
        self.seq = self.seq + 1

    def get_seq(self):
        return self.seq

    def set_batterbox(self, batterbox):
        self.batterbox = batterbox

    def get_batterbox(self):
        return self.batterbox

    def set_inn(self, inn):
        self.inn = inn

    def get_inn(self):
        return self.inn

    #def set_LineUp(self, LineUp):

