import threading

from Video.video import play
from Video.tts import TTS
from Web.web import Web
from Vision.vision import Vision
from resource import Resource

class KBO():
    def __init__(self, isSimulation=0):
        self.isSimulation = isSimulation

        self.resource = Resource()
        self.web = Web(resource=self.resource)
        self.vision = Vision(resource=self.resource, isSimulation=self.isSimulation)
        self.tts = TTS(resource=self.resource)


    def run(self):
        web_thread = threading.Thread(target=self.web.parsing_relaytext)
        web_thread.start()

        vision_thread = threading.Thread(target=self.vision.play)
        vision_thread.start()

        #tts = threading.Thread(target=self.tts.text_2_speech)
        #tts.start()

        play(self.resource, self.isSimulation)

if __name__ == '__main__':
    app = KBO()
    app.run()