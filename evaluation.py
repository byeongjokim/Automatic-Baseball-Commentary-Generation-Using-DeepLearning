from NN.detect_model import Detect_Model
from NN.motion_model import Classifier
import tensorflow as tf

def detect_model():
    sess = tf.Session()
    a = Detect_Model(sess, 1)
    a.evaluation()

def motion_model():
    sess = tf.Session()
    a = Classifier(sess, 1)
    a.evaluation()

motion_model()
#detect_model()