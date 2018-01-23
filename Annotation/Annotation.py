import time
from Annotation.Naver_data import NaverData
from Annotation.Scene_data import SceneData

class Annotation():
    frame_no = 0

    def __init__(self, fileName, Resources):
        print("init annotation")
        self.Resources = Resources

        self.naverData = NaverData(fileName, Resources)
        self.sceneData = SceneData(Resources)

    def generate_Naver(self, count_delta, fps, o_start):
        start, no = self.naverData.calculate_start(count_delta, fps, o_start)
        self.naverData.return_seq(start, no)

    def generate_Scene(self):
        while(not self.Resources.exit):

            if(self.Resources.long==True):
                #self.sceneData.predict(self.Resources.frame_no, self.Resources.now_relayText)
                self.sceneData.predict_with_frame(self.Resources.frame, self.Resources.now_relayText)
                self.Resources.set_long(False)
            time.sleep(7)
            self.Resources.set_long(True)

