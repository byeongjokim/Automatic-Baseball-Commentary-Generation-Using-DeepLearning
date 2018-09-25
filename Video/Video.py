import cv2
import numpy as np
from time import time as timer
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

        # cv2.namedWindow('play', cv2.WINDOW_NORMAL)
        fps = video.get(cv2.CAP_PROP_FPS)

        print(fps)
        fps /= 1000

        while True:
            start = timer()

            success, frame = video.read()
            if not success:
                self.Resources.set_exit(True)
                break
            if cv2.waitKey(1) == ord('q'):
                self.Resources.set_exit(True)
                break
            if cv2.waitKey(1) == ord('a'):
                print("real ", str(count))

            self.Resources.set_frame(frame)

            textimage = self.text_2_img(self.Resources.get_annotation())
            frame[h-100:h-50, 100:w-100] = textimage
            cv2.imshow('play', frame)

            diff = timer() - start
            while diff < fps:
                diff = timer() - start

            count = count + 1

    @staticmethod
    def text_2_img(text):
        img = Image.new('RGB', (1080, 50), color=(180, 180, 180))
        font = ImageFont.truetype("gulim.ttc", 20)
        d = ImageDraw.Draw(img)
        d.text((10, 10), text, font=font, fill=(0, 0, 0))

        img.save('pil_text_font.png')
        img = np.asarray(img)

        return img


cv2.destroyAllWindows()
