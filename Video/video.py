import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from time import time
import settings

def play(resource, isSimulation):
    def text_2_img(text):
        img = Image.new('RGB', (1080, 50), color=(180, 180, 180))
        font = ImageFont.truetype("gulim.ttc", 20)
        #font = ImageFont.truetype("gulim.ttc", 30)
        d = ImageDraw.Draw(img)
        d.text((10, 10), text, font=font, fill=(0, 0, 0))

        return np.asarray(img)

    video = cv2.VideoCapture(settings.VIDEO_FILE)
    video.set(1, settings.START_FRAME)
    fps = 1 / 29.97

    success, frame = video.read()
    h, w, c = frame.shape
    frameno = settings.START_FRAME + 1
    resource.set_frameno(frameno)

    text = resource.get_annotation()
    textimage = text_2_img(text)

    while (cv2.waitKey(1) != ord('q')):
        start = time()
        frameno = frameno + 1

        success, frame = video.read()

        resource.set_frame(frame)
        resource.set_frameno(frameno)

        if (resource.is_new_annotation_video()):
            text = resource.get_annotation()
            textimage = text_2_img(text)

        frame[h - 100 : h - 50, 100 : w - 100] = textimage
        cv2.imshow("Automatic Sports Commentary", frame)

        diff = time() - start
        while diff < fps:
            diff = time() - start