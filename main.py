import threading
import cv2
from Video.Video import Video
from Annotation.Annotation import Annotation
from resources import Resources


resources = Resources()

annotation = Annotation('./_data/20171030KIADUSAN/20171030KIADUSAN.txt', resources)
video = Video(resources)

o_start = "183122"
o_count = 8145
fps = 29.97

#count = 70233  before start 2
#count = 10000
count = 165050
count = 9500 #3

count = 900

naver = threading.Thread(target=annotation.generate_Naver, args=(count-o_count, fps, o_start, ))
naver.start()

scene = threading.Thread(target=annotation.generate_Scene)
scene.start()

v = "20171030KIADUSAN"
#v= "180407LGLT"
v= "180407NCDUSAN"

#6ghl
video.play(v="./_data/"+v+"/"+v+".mp4", count=count)
