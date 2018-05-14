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

        self.naverData = NaverData(gameName, Resources)
        self.sceneData = SceneData(Resources)

    def generate_Naver(self, count_delta, fps, o_start):
        self.naverData.get_relayText()

    def generate_Scene(self):
        count = 0
        frame = []
        flag = 0
        label = -1

        while(not self.Resources.exit):
            if len(frame) == 0 or count == 0:
                frame = self.Resources.frame
                count = count + 1
            else:
                if(ssim(self.Resources.frame, frame, multichannel=True) < 0.6):
                    flag = 0
                    frame = self.Resources.frame
                    #print("\t\tscene changed")
                    label = self.sceneData.Annotation_with_frame(self.Resources.frame, self.Resources.now_relayText)

                else:
                    if(flag == 15 and label == 2):
                        print("투수입니다")
                    flag = flag + 1



















    def Rule_Base_String_Maker(self, text):
        if("스트라이크" in text):
            return 1



    def get_ball_location(self):
        return 1

    def strikeout(self, name):
        annotaion = [
            "스트라이크 아웃! " + name + " 선수는 삼진으로 물러납니다.",
            "삼진! 결국 "+ name + " 선수를 삼진으로 잡아내는군요.",
            "삼진~! " + name + " 선수를 삼진 처리하면서 아웃카운트 하나를 따내는군요."
        ]

        return random.choice(annotaion)

    def baseOnBalls(self, name):
        annotaion = [
            name + " 선수는 볼넷으로 걸어나갑니다.",
            name + " 선수, 끈질긴 승부 끝에 볼넷을 얻어냅니다."
        ]

        return random.choice(annotaion)

    def intentionalBaseOnBalls(self, name):
        annotaion = [
            name + " 선수는 고의사구로 걸어나갑니다.",
            "배터리가 " + name + " 선수를 거르는군요. " + name + " 선수, 고의사구로 출루합니다."
        ]

        return random.choice(annotaion)

    def hitByPitch(self, name):
        annotaion = [
            "아, 몸에 맞았군요. " + name + " 선수 몸에 맞는 볼로 출루합니다.",
            name + " 선수 몸에 맞았어요. 몸에 맞는 볼로 걸어 나가네요."
        ]

        return random.choice(annotaion)

    def outfieldSingleHit(self, name, pos):
        if ("중간" in pos):
            return self.centerSingleHit(name, pos);
        else:
            return self.frontPosSingleHit(name, pos);

    def centerSingleHit(self, name, pos):
        annotaion = [
            pos + "에 떨어지는 안타 만들어내면서 " + name + " 선수 1루에 안착합니다."
        ]

        return random.choice(annotaion)

    def frontPosSingleHit(self, name, pos):
        annotaion = [
            pos + " 앞에 떨어지는 안타! " + name + " 선수 안타 만들어냅니다."
        ]

        return random.choice(annotaion)
