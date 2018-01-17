import threading
from Video import Video
from Annotation.Annotation import Annotation

annotation = Annotation('./_data/20171030KIADUSAN.txt')
video = Video(annotation)

naver = threading.Thread(target=annotation.generate_Naver)
naver.start()

scene = threading.Thread(target=annotation.generate_Scene)
scene.start()

video.play(v="./_data/20171030KIADUSAN.mp4")
