from NN.tinyYOLOv2.test import ObjectDetect
import cv2
import tensorflow as tf

def test():
    image = cv2.imread("./_data/180407HHKT/233.jpg")
    h, w, c = image.shape
    sess = tf.Session()
    objectdetect = ObjectDetect(sess)
    bboxes = objectdetect.predict(image)

    ratio_h = h / 416
    ratio_w = w / 416
    for bbox in bboxes:
        b = image[int(bbox[0][1]*ratio_h) : int(bbox[0][3]*ratio_h), int(bbox[0][0]*ratio_w) : int(bbox[0][2]*ratio_w)]

        cv2.imshow("a", b)
        cv2.waitKey(0)

test()
