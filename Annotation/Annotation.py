import time
import random
from Annotation.Naver_data import NaverData
from Annotation.Scene_data import SceneData

from skimage.measure import compare_ssim as ssim

class Annotation():
    frame_no = 0
    frame = []

    def __init__(self, gameName, Resources):
        print("init annotation")
        self.Resources = Resources

        self.naverData = NaverData(gameName)
        self.sceneData = SceneData(Resources)

    def generate_Naver(self, count_delta, fps, o_start):
        #self.naverData.set_Start(count_delta, fps, o_start)
        self.naverData.get_Annotation()

    def generate_Scene(self):
        return 1