import threading
import cv2

import argparse

from Video.Video import Video
from Annotation.Middle import Middle
from resources import Resources

v = "180515SKOB_FULL"

#fps = cv2.VideoCapture("./_data/"+v+"/"+v+".mp4").get(cv2.CAP_PROP_FPS)
#print(fps)
resources = Resources()
annotation = Middle(v, resources)
video = Video(resources)

o_start = "183112"
o_count = 20800
fps = 29.97

count = 123180

rule = threading.Thread(target=annotation.generate_Annotation_with_Rule, args=(count-o_count, fps, o_start, ))
rule.start()

#scene = threading.Thread(target=annotation.generate_Annotation_with_Scene)
#scene.start()

video.play(v="./_data/"+v+"/"+v+".mp4", count=count)
