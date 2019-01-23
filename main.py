import threading
import cv2

import argparse

from Video.Video import Video
from Video.tts import TTS
from Annotation.Middle import Middle
from resources import Resources

v = "180906LGNC_FULL"
#v = "180515SKOB_FULL"

#fps = cv2.VideoCapture("./_data/"+v+"/"+v+".mp4").get(cv2.CAP_PROP_FPS)
#print(fps)
resources = Resources()
annotation = Middle(v, resources)
video = Video(resources)
tts = TTS(resources)

o_start = "183013"
o_count = 9600
fps = 29.97

#count = 66300
#count = 28780
#count = 26300

#count = 91000
#count = 52000

#count = 75000
#count = 77000

#count = 78500
#count = 80000
#count = 81000


#count = 58060 #외야 안타
#count = 60720 #더블플레이
#count = 21528 #1루땅볼
count = 126500


#count = 60000
#count = 300000

rule = threading.Thread(target=annotation.generate_Annotation_with_Rule, args=(count-o_count, fps, o_start, ))
rule.start()

scene = threading.Thread(target=annotation.generate_Annotation_with_Scene)
scene.start()

tts = threading.Thread(target=tts.text_2_speech)
tts.start()

video.play(v="./_data/"+v+"/"+v+".mp4", count=count, fps=fps)
