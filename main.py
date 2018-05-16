import threading
import argparse

from Video.Video import Video
from Annotation.Annotation import Annotation
from resources import Resources

resources = Resources()

v = "20171030HTOB"

annotation = Annotation('180501LGHH', resources)
video = Video(resources)

o_start = "183122"
o_count = 5800
fps = 29.97

count = 10000

naver = threading.Thread(target=annotation.generate_Naver, args=(count-o_count, fps, o_start, ))
naver.start()

scene = threading.Thread(target=annotation.generate_Scene)
scene.start()

video.play(v="./_data/"+v+"/"+v+".mp4", count=count)
