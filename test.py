from NN.tinyYOLOv2.test import ObjectDetect
import cv2
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow



checkpoint_path = './NN/tinyYOLOv2/ckpt/model.ckpt'

reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    print("tensor_name: ", key)


print("adasdas")


checkpoint_path = '_model/scene/scene.ckpt'
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)

checkpoint_path = '_model/motion/motion.ckpt'
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key).shape)

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
