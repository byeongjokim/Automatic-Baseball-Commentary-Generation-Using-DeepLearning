import cv2
import numpy as np
from time import time
from PIL import Image, ImageDraw, ImageFont

class Video():
    def __init__(self, Resources):
        print("init_video")
        self.Resources = Resources


    def play(self, v, count, fps):
        video = cv2.VideoCapture(v)

        video.set(1, count)

        success, frame = video.read()
        if (success):
            h, w, c = frame.shape

        fps = 1/fps
        textimage = self.text_2_img(self.Resources.get_annotation())
        while True:
            start = time()

            success, frame = video.read()
            if not success:
                self.Resources.set_exit(True)
                break
            if cv2.waitKey(1) == ord('q'):
                self.Resources.set_exit(True)
                break

            self.Resources.set_frame(frame)

            if(self.Resources.is_new_annotation_text()):
                text = self.Resources.get_annotation()
                textimage = self.text_2_img(text)

            frame[h-100:h-50, 100:w-100] = textimage

            cv2.imshow('play', frame)

            diff = time() - start
            while diff < fps:
                diff = time() - start

    def text_2_img(self, text):
        img = Image.new('RGB', (1080, 50), color=(180, 180, 180))
        font = ImageFont.truetype("gulim.ttc", 20)
        #font = ImageFont.truetype("gulim.ttc", 30)
        d = ImageDraw.Draw(img)
        d.text((10, 10), text, font=font, fill=(0, 0, 0))

        return np.asarray(img)


cv2.destroyAllWindows()
