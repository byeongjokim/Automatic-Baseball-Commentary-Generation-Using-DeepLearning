import threading

from Video.video import play, play_bbox
from Video.tts import TTS
from Web.web import Web
from Vision.vision import Vision
from resource import Resource

class KBO():
    def __init__(self, isSimulation=False):
        if not(isSimulation):
            self.resource = Resource()
            self.web = Web(resource=self.resource)
            self.tts = TTS(resource=self.resource)
            self.vision = Vision(resource=self.resource)
        else:
            print("====================Simulation====================")
            

    def run(self):
        web_thread = threading.Thread(target=self.web.parsing_relaytext)
        web_thread.start()

        vision_thread = threading.Thread(target=self.vision.play)
        vision_thread.start()

        #tts = threading.Thread(target=self.tts.text_2_speech)
        #tts.start()

        play(self.resource)
    
    def run_bbox(self):
        play_bbox(frameno=128400)

if __name__ == '__main__':
    app = KBO(isSimulation=True)
    #app.run()
    app.run_bbox()