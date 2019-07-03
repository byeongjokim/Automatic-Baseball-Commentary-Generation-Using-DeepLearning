import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from time import time
import settings
import csv

def play(resource):
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
    frameno = settings.START_FRAME + 1
    if(success):
        h, w, c = frame.shape

    text = resource.get_annotation()
    #textimage = text_2_img(text)
    resource.set_frameno(settings.START_FRAME + 1)

    #result_text = open("./_result4/" + settings.FILE_NAME + "/result.csv", "w")
    #writer = csv.writer(result_text, delimiter=",")
    #save_flag = 0
    #save_imgno = 0

    while True:
        start = time()

        success, frame = video.read()
        frameno = frameno + 1
        resource.set_frame(frame)
        resource.set_frameno(frameno)
        if cv2.waitKey(1) == ord('q'):
            break

        if (resource.is_new_annotation_video()):
            save_flag = 1
            text = resource.get_annotation()
            #textimage = text_2_img(text)

        #frame[h - 100 : h - 50, 100 : w - 100] = textimage
        cv2.imshow("play", frame)

        """
        if(save_flag == 1):
            if(frameno > 171332):
                print("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
                result_text.close()
                break
            cv2.imwrite("./_result4/" + settings.FILE_NAME + "/" + str(frameno).zfill(3) + ".jpg", frame)
            writer.writerow([str(frameno), text])
            save_flag = 0
            save_imgno = save_imgno + 1
        """

        diff = time() - start
        while diff < fps:
            diff = time() - start