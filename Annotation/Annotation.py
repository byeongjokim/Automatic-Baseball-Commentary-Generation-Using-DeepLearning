import time
from Annotation.Naver_data import NaverData
from Annotation.Scene_data import SceneData

class Annotation():
    frame_no = 0

    def __init__(self, fileName):
        print("init annotation")
        self.naverData = NaverData(fileName)
        self.sceneData = SceneData()

    def generate_Naver(self):
        self.naverData.return_seq()

    def generate_Scene(self):
        while(True):

            if(self.naverData.long==True):
                self.sceneData.predict(self.frame_no, self.naverData.now_relayText)
                self.naverData.long = False
            time.sleep(7)
            self.naverData.long = True



    def set_frameNo(self, frame_no):
        self.frame_no = frame_no
