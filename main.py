import cv2
import threading
from data import TextData, SceneData

textData = TextData()
sceneData = SceneData("./_data/m.csv")
def video():
    video = cv2.VideoCapture("./_data/20171030KIADUSAN.mp4")
    count = 8100
    video.set(1, count)
    while True:
        success, frame = video.read()
        if not success:
            break

        cv2.imshow("a", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()



#textData.return_seq()
#t.start()


