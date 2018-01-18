import cv2

class Video():
    def __init__(self, Resources):
        print("init_video")
        self.Resources = Resources


    def play(self, v, count):
        video = cv2.VideoCapture(v)

        video.set(1, count)

        while True:
            success, frame = video.read()
            if not success:
                self.Resources.set_exit(True)
                break

            cv2.imshow("1030KIADS", frame)

            self.Resources.set_frameNo(count)

            if cv2.waitKey(1) == ord('q'):
                self.Resources.set_exit(True)
                break
            if cv2.waitKey(1) == ord('a'):
                print("real ", str(count))

            count = count + 1

cv2.destroyAllWindows()
