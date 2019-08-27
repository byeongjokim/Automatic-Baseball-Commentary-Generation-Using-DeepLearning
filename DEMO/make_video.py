import cv2
import glob
import numpy as np
def images_to_video(path='./with_bbox_labels/'):
    img_array = []

    for filename in glob.glob(path + '*.jpg'):
        img = cv2.imread(filename)
        img = cv2.resize(img, (1280, 720), cv2.INTER_LINEAR)
        img = np.array(img)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(path + 'video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
    print(len(img_array))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def video_to_imges(v='./DEMO_ENG.mp4'):
    video = cv2.VideoCapture(v)
    success, frame = video.read()
    cv2.imwrite("./DEMO_ENG_FIRST.jpg", frame)

images_to_video()
#video_to_imges()