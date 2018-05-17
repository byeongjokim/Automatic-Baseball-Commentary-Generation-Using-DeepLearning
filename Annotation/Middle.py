import time
import random
import numpy as np
from Annotation.Rule_Annotation import RuleData
from Annotation.Scene_Annotation import SceneData

from skimage.measure import compare_ssim as ssim

class Middle():
    frame_no = 0
    frame = []

    def __init__(self, gameName, Resources):
        print("init annotation")
        self.Resources = Resources

        self.ruleData = RuleData(gameName)
        self.sceneData = SceneData(Resources)

    def generate_Annotation_with_Rule(self, count_delta, fps, o_start):
        self.ruleData.set_Start(count_delta, fps, o_start)
        self.ruleData.get_Annotation()
        return 1

    def generate_Annotation_with_Scene(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        isPitcher = 0

        time.sleep(3)
        print("Scene model 시작")
        label = -1
        while( not self.Resources.exit):
            if(ssim(self.Resources.frame, frame, multichannel=True) < 0.6):
                isPitcher = 0
                frame = self.Resources.frame
                label = self.sceneData.get_Annotation(self.Resources.frame)
            else:
                if(isPitcher == 15 and label == 2):
                    print("이 투수는 ~~~")
                isPitcher = isPitcher + 1


        return 1