import cv2
from time import time as timer

class Video():
    def __init__(self, Resources):
        print("init_video")
        self.Resources = Resources


    def play(self, v, count, fps):
        video = cv2.VideoCapture(v)

        video.set(1, count)
        cv2.namedWindow('play', cv2.WINDOW_NORMAL)
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
            cv2.imshow('play', self.Resources.frame)

            diff = timer() - start
            while diff < fps:
                diff = timer() - start

            count = count + 1

cv2.destroyAllWindows()
