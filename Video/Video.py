import cv2

class Video():
    def __init__(self, Resources):
        print("init_video")
        self.Resources = Resources


    def play(self, v, count, fps):
        video = cv2.VideoCapture(v)

        video.set(1, count)
        cv2.namedWindow('play', cv2.WINDOW_NORMAL)

        while True:
            success, frame = video.read()
            if not success:
                self.Resources.set_exit(True)
                break

            #output = cv2.resize(frame, (960,1080))


            #self.Resources.set_frameNo(count)
            self.Resources.set_frame(frame)
            cv2.imshow('play', self.Resources.frame)

            if cv2.waitKey(1) == ord('q'):
                self.Resources.set_exit(True)
                break
            if cv2.waitKey(1) == ord('a'):
                print("real ", str(count))

            count = count + 1

cv2.destroyAllWindows()
