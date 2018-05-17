import threading
import argparse

from Video.Video import Video
from Annotation.Middle import Middle
from resources import Resources

resources = Resources()

v = "20171030HTOB"

annotation = Middle(v, resources)
video = Video(resources)

o_start = "183122"
o_count = 8155
fps = 29.97

count = 100000

rule = threading.Thread(target=annotation.generate_Annotation_with_Rule, args=(count-o_count, fps, o_start, ))
rule.start()

scene = threading.Thread(target=annotation.generate_Annotation_with_Scene)
scene.start()

video.play(v="./_data/"+v+"/"+v+".mp4", count=count)
