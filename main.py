import threading
import cv2
from Video.Video import Video
from Annotation.Annotation import Annotation
from Person.Person import Person
from resources import Resources


resources = Resources()

annotation = Annotation('./_data/20171030KIADUSAN/20171030KIADUSAN.txt', resources)
person = Person(annotation, resources)
video = Video(resources)

o_start = "183122"
o_count = 8145
fps = 29.97

#count = 70233  before start 2
#count = 10000
count = 55000

naver = threading.Thread(target=annotation.generate_Naver, args=(count-o_count, fps, o_start, ))
naver.start()

scene = threading.Thread(target=annotation.generate_Scene)
scene.start()

video.play(v="./_data/20171028KIADUSAN/20171028KIADUSAN.mp4", count=count)
#video.play(v="./_data/20171029KIADUSAN/20171029KIADUSAN.mp4", count=count)
#video.play(v="./_data/20171028KIADUSAN_HIGHLIGHT.mp4", count=count)
