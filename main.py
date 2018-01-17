import cv2
import threading
import time
from data import TextData, SceneData

#sceneData = SceneData("./_data/scene1-1.csv")
#sceneData.load_image_data()



def video():
    video = cv2.VideoCapture("./_data/20171030KIADUSAN.mp4")
    count = 8150
    #count = 8607
    video.set(1, count)
    fps = video.get(cv2.CAP_PROP_FPS)

    while True:
        success, frame = video.read()
        if not success:
            break

        cv2.imshow("a", frame)


        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.waitKey(1) == ord('a'):
            print("real " , str(count))
        count = count + 1


    cv2.destroyAllWindows()



lock = threading.Lock()

video = cv2.VideoCapture("./_data/20171030KIADUSAN.mp4")
count = 8150
# count = 8607
video.set(1, count)
#fps = video.get(cv2.CAP_PROP_FPS)


textData = TextData()
t = threading.Thread(target=textData.return_seq)
t.start()

while True:
    success, frame = video.read()
    if not success:
        break

    textData.setframeNo(count)

    cv2.imshow("1030KIADS", frame)

    if cv2.waitKey(1) == ord('q'):
        break
    if cv2.waitKey(1) == ord('a'):
        print("real ", str(count))
    count = count + 1

cv2.destroyAllWindows()




