import cv2
import glob
import numpy as np
def images_to_video(path='./with_bbox_labels/'):
    img_array = []

    for filename in glob.glob(path + '*.jpg'):
        img = cv2.imread(filename)
        img = np.array(img)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(path + 'video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

images_to_video()