class Resources():
    long = 0
    frame_no = 0
    frame = []
    exit = False
    now_relayText = ''

    motion_weight = 60
    motion_height = 80


    def set_frameNo(self, frame_no):
        self.frame_no = frame_no

    def set_frame(self, frame):
        self.frame = frame

    def set_long(self, long):
        self.long = long

    def set_now_relayText(self, now_relayText):
        self.now_relayText = now_relayText

    def set_exit(self, exit):
        self.exit = exit
