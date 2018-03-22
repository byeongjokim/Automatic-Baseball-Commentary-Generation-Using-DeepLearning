import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Person_model():

    width = 224
    height = 224

    max_num = 10

    cnn_batch_size = 20
    cnn_chk = './_model/person/cnn/cnn.ckpt'
    cnn_ckpt = tf.train.get_checkpoint_state(("./_model/person/cnn/"))


    def __init__(self):
        print("init person model")

    def load_data(self):
        mnist = input_data.read_data_sets("_data/MNIST_data", one_hot=True)

        print("use " + str(mnist.train.num_examples) + " Mnist Data")
        return 1

    def pre_train_CNN(self):
        leraning_rate = 0.001
        epoch = 15
        batch_size = 100


        return 1

    def make_model(self):
        return 1

    def train(self):
        return 1

    def test(self):
        return 1

    def predict(self, image):
        return 1

