import threading
from Video import Video
from Annotation.Annotation import Annotation

annotation = Annotation('./_data/20171030KIADUSAN.txt')
video = Video(annotation)

o_start = "183122"
o_count = 8145
count = 7000
fps = 29.97

naver = threading.Thread(target=annotation.generate_Naver, args=(count-o_count, fps, o_start, ))
naver.start()

scene = threading.Thread(target=annotation.generate_Scene)
scene.start()

video.play(v="./_data/20171030KIADUSAN.mp4", count=count)


