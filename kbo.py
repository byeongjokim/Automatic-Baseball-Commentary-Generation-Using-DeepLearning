import threading
import socket

from settings import START_FRAME
from Video.video import play, play_bbox, play_API
from Video.tts import TTS
from Web.web import Web
from Vision.vision import Vision
from resource import Resource

class KBO():
    def __init__(self, isSimulation=False, isAPI=False):
        if isAPI:
            print("====================As API====================")
            self.resource = Resource()
            self.web = Web(resource=self.resource)
            self.vision = Vision(resource=self.resource)

            self.run_server(host="166.104.143.103", port=8080)

        elif isSimulation:
            print("====================Simulation====================")
            self.run_bbox()

        else:
            print("====================In Local====================")
            self.resource = Resource()
            self.web = Web(resource=self.resource)
            self.vision = Vision(resource=self.resource)
            self.tts = TTS(resource=self.resource)

            self.run()

    def run(self):
        self.resource.set_frameno(START_FRAME + 1)
        idx = self.web.parsing_before()

        web_thread = threading.Thread(target=self.web.parsing_relaytext, args=(idx,))
        web_thread.start()

        vision_thread = threading.Thread(target=self.vision.play)
        vision_thread.start()

        tts = threading.Thread(target=self.tts.text_2_speech)
        tts.start()

        play(self.resource)
    
    def run_bbox(self):
        play_bbox(frameno=128400)
    
    def run_server(self, host, port):
        self.resource.set_frameno(START_FRAME + 1)
        idx = self.web.parsing_before()

        web_thread = threading.Thread(target=self.web.parsing_relaytext, args=(idx, True))
        web_thread.start()

        vision_thread = threading.Thread(target=self.vision.play)
        vision_thread.start()

        video_thread = threading.Thread(target=play_API, args=(self.resource,))
        video_thread.start()

        with socket.socket() as sock:
            sock.connect((host, port))

            while(True):
                if (self.resource.is_new_annotation_video()):
                    text = self.resource.get_annotation()
                    print(text)
                    sock.sendall(text.encode())



if __name__ == '__main__':
    app = KBO(isSimulation=False, isAPI=True)