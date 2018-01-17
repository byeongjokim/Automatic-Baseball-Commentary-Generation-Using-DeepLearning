import cv2

class Video():
    def __init__(self, Annotation):
        print("init_video")
        self.Annotation = Annotation

    def play(self, v):
        video = cv2.VideoCapture(v)
        count = 8150
        # count = 8607

        video.set(1, count)

        while True:
            success, frame = video.read()
            if not success:
                break

            cv2.imshow("1030KIADS", frame)

            self.Annotation.set_frameNo(count)
            #self.Annotation.set_frame(frame)

            if cv2.waitKey(1) == ord('q'):
                break
            if cv2.waitKey(1) == ord('a'):
                print("real ", str(count))
            count = count + 1

        cv2.destroyAllWindows()