import threading
from Video.Video import Video
from Annotation.Annotation import Annotation
from Person.Person import Person
from resources import Resources
from Annotation.Scene_data import Make_SceneData

'''
s = Make_SceneData('./_data/scene1-1.csv')
s.clustering()
s.save_image_data()
'''

resources = Resources()

annotation = Annotation('./_data/20171030KIADUSAN.txt', resources)
person = Person(annotation, resources)
video = Video(resources)

o_start = "183122"
o_count = 8145
fps = 29.97

count = 10000

naver = threading.Thread(target=annotation.generate_Naver, args=(count-o_count, fps, o_start, ))
naver.start()

scene = threading.Thread(target=annotation.generate_Scene)
scene.start()

video.play(v="./_data/20171030KIADUSAN.mp4", count=count)
