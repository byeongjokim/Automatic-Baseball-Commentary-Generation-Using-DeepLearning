import cv2
import tensorflow as tf
import numpy as np
import os
import random
import csv
from NN.cnn import conv_layer, pool

class Scene_Model():
    chk_scene = './_model/scene/scene.ckpt'
    ckpt = tf.train.get_checkpoint_state(("./_model/scene"))

    batch_size = 30
    epoch = 100

    width = 224
    height = 224

    #kind_scene = ['field', 'pitcher', 'gallery', 'batter', 'pitchingbatting', 'beforestart', '1', 'coach', 'closeup', '3']
    kind_scene = ["pitchingbatting", "1", "batter", "closeup", "coach", "gallery", "field", "etc", "3"]
    num_label = len(kind_scene)

    def __init__(self):
        print("init scene_model")

    def load_data(self):
        csv_ = ["./_data/20171029KIADUSAN.csv", "./_data/20171030KIADUSAN.csv"]
        folder_ = ["./_data/20171029KIADUSAN/", "./_data/20171030KIADUSAN/"]

        play = ["20171029KIADUSAN", "20171030KIADUSAN"]

        dataset=[]
        data_set = []

        for p in play:
            folder_path = "./_data/" + p + "/"
            csv_path = folder_path + p + ".csv"
            print(csv_path)

            f = open(csv_path, "r")
            reader = csv.reader(f)
            for line in reader:
                sett = {"start":line[0], "end":line[1], "label":line[2]}
                dataset.append(sett)
            f.close()


            for i in dataset:
                for j in range(int(i["start"]), int(i["end"])+1, 7):
                    image = cv2.resize(
                        cv2.cvtColor(
                            cv2.imread(folder_path + str(j) + ".jpg"),
                            cv2.COLOR_BGR2GRAY
                        ),
                        (self.width, self.height)
                    )
                    data_set.append({"image":image, "label":int(i["label"])})

        self.X = np.array([i["image"] for i in data_set])
        _y = np.array([i["label"] for i in data_set])
        self.Y = np.zeros((len(_y), len(set(_y))))
        self.Y[np.arange(len(_y)), [i-1 for i in _y]] = 1

        print(self.X.shape)
        print(self.Y.shape)


        '''
        path = "./scene_data/test/"
        image = []
        for (p, dir, files) in os.walk(path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.jpg':
                    image.append(filename)

        self.data = []
        for i in image:
            self.data.append({"image":
                cv2.resize(
                    cv2.cvtColor(
                        cv2.imread(path + i),
                        cv2.COLOR_BGR2GRAY),
                    (self.width, self.height)),
                "label": i.split(".")[0].split("_")[0]})


        random.shuffle(self.data)


        x = np.array([i["image"] for i in self.data])
        _y = np.array([i["label"] for i in self.data])
        y = np.zeros((len(_y), len(set(_y))))
        y[np.arange(len(_y)), [self.kind_scene.index(i) for i in _y]] = 1

        self.X = x
        self.Y = y
        '''


    #vggnet - A
    def make_model(self):
        """
        self.scene_X = tf.placeholder(tf.float32, [None, self.width, self.height, 1])
        self.scene_Y = tf.placeholder(tf.float32, [None, self.num_label])
        self.scene_keep_prob = tf.placeholder(tf.float32)

        C1_1 = conv_layer(filter_size=3, fin=1, fout=64, din=self.scene_X, name='scene_C1_1')
        C1_2 = conv_layer(filter_size=3, fin=64, fout=64, din=C1_1, name='scene_C1_2')
        P1 = pool(C1_2, option="maxpool")

        C2_1 = conv_layer(filter_size=3, fin=64, fout=128, din=P1, name='scene_C2_1')
        C2_2 = conv_layer(filter_size=3, fin=128, fout=128, din=C2_1, name='scene_C2_2')
        P2 = pool(C2_2, option="maxpool")

        C3_1 = conv_layer(filter_size=3, fin=128, fout=256, din=P2, name='scene_C3_1')
        C3_2 = conv_layer(filter_size=3, fin=256, fout=256, din=C3_1, name='scene_C3_2')
        C3_3 = conv_layer(filter_size=1, fin=256, fout=256, din=C3_2, name='mscene_C3_3')
        P3 = pool(C3_3, option="maxpool")

        C4_1 = conv_layer(filter_size=3, fin=256, fout=512, din=P3, name='scene_C4_1')
        C4_2 = conv_layer(filter_size=3, fin=512, fout=512, din=C4_1, name='scene_C4_2')
        C4_3 = conv_layer(filter_size=1, fin=512, fout=512, din=C4_2, name='scene_C4_3')
        P4 = pool(C4_3, option="maxpool")

        C5_1 = conv_layer(filter_size=3, fin=512, fout=512, din=P4, name='scene_C5_1')
        C5_2 = conv_layer(filter_size=3, fin=512, fout=512, din=C5_1, name='scene_C5_2')
        C5_3 = conv_layer(filter_size=1, fin=512, fout=512, din=C5_2, name='mscene_C5_3')
        P5 = pool(C5_3, option="maxpool")

        print(P5)

        fc0 = tf.reshape(P5, [-1, 7 * 7 * 512])

        with tf.device("/cpu:0"):
            W1 = tf.get_variable("scene_W1", shape=[7 * 7 * 512, 4096],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([4096]))
            fc1 = tf.nn.relu(tf.matmul(fc0, W1) + b1)

            W2 = tf.get_variable("model_W2", shape=[4096, 4096], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([4096]))
            fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)

            W3 = tf.get_variable("model_W3", shape=[4096, self.num_label], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([self.num_label]))
            self.scene_model = tf.matmul(fc2, W3) + b3
        """
        self.scene_X = tf.placeholder(tf.float32, [None, self.width, self.height, 1])
        self.scene_Y = tf.placeholder(tf.float32, [None, self.num_label])
        self.scene_keep_prob = tf.placeholder(tf.float32)

        C1 = conv_layer(filter_size=3, fin=1, fout=64, din=self.scene_X, name='scene_C1')
        P1 = pool(C1, option="maxpool")
        P1 = tf.nn.dropout(P1, keep_prob=self.scene_keep_prob)

        C2 = conv_layer(filter_size=3, fin=64, fout=128, din=P1, name='scene_C2')
        P2 = pool(C2, option="maxpool")
        P2 = tf.nn.dropout(P2, keep_prob=self.scene_keep_prob)

        C3_1 = conv_layer(filter_size=3, fin=128, fout=256, din=P2, name='scene_C3_1')
        C3_2 = conv_layer(filter_size=3, fin=256, fout=256, din=C3_1, name='scene_C3_2')
        P3 = pool(C3_2, option="maxpool")
        P3 = tf.nn.dropout(P3, keep_prob=self.scene_keep_prob)

        C4_1 = conv_layer(filter_size=3, fin=256, fout=512, din=P3, name='scene_C4_1')
        C4_2 = conv_layer(filter_size=3, fin=512, fout=512, din=C4_1, name='scene_C4_2')
        P4 = pool(C4_2, option="maxpool")
        P4 = tf.nn.dropout(P4, keep_prob=self.scene_keep_prob)

        C5_1 = conv_layer(filter_size=3, fin=512, fout=512, din=P4, name='scene_C5_1')
        C5_2 = conv_layer(filter_size=3, fin=512, fout=512, din=C5_1, name='scene_C5_2')
        P5 = pool(C5_2, option="maxpool")
        P5 = tf.nn.dropout(P5, keep_prob=self.scene_keep_prob)

        print(P5)

        fc0 = tf.reshape(P5, [-1, 7 * 7 * 512])

        with tf.device("/cpu:0"):
            W1 = tf.get_variable("scene_W1", shape=[7 * 7 * 512, 4096],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([4096]))
            fc1 = tf.nn.relu(tf.matmul(fc0, W1) + b1)

            W2 = tf.get_variable("scene_W2", shape=[4096, 1000], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([1000]))
            fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)

            W3 = tf.get_variable("scene_W3", shape=[1000, self.num_label],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([self.num_label]))
            self.scene_model = tf.matmul(fc2, W3) + b3

        print(self.scene_model)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scene_model, labels=self.scene_Y))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        if self.ckpt and tf.train.checkpoint_exists(self.ckpt.model_checkpoint_path):
            print("rstore the sess!!")
            self.saver.restore(self.sess, self.chk_scene)
        else:
            self.sess.run(tf.global_variables_initializer())


    def train(self):
        train_x = self.X[:-10]
        train_y = self.Y[:-10]

        total_batch = int(len(train_x) / self.batch_size)

        if (total_batch == 0):
            total_batch = 1

        for e in range(self.epoch):
            total_cost = 0

            j = 0
            for i in range(total_batch):
                if (j + self.batch_size > len(train_x)):
                    batch_x = train_x[j:]
                    batch_y = train_y[j:]
                else:
                    batch_x = train_x[j:j + self.batch_size]
                    batch_y = train_y[j:j + self.batch_size]
                    j = j + self.batch_size

                batch_x = batch_x.reshape(-1, self.width, self.height, 1)
                batch_y = batch_y.reshape(-1, self.num_label)

                _, cost_val = self.sess.run([self.optimizer, self.cost],
                                            feed_dict={self.scene_X: batch_x, self.scene_Y: batch_y, self.scene_keep_prob: 0.8})

                total_cost = total_cost + cost_val

            if (total_cost / total_batch < 0.03):
                break
            print('Epoch:', '%d' % (e + 1), 'Average cost =', '{:.3f}'.format(total_cost / total_batch))

        print("complete")
        self.saver.save(self.sess, self.chk_scene)

    def test(self):
        test_x = self.X[-10:]
        test_y = self.Y[-10:]

        is_correct = tf.equal(tf.argmax(self.scene_model, 1), tf.argmax(self.scene_Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        test_x = test_x.reshape(-1, self.width, self.height, 1)
        print('accuracy: ',
              self.sess.run(accuracy, feed_dict={self.scene_X: test_x, self.scene_Y: test_y, self.scene_keep_prob: 1}) * 100)

        print("Label: ", self.sess.run(tf.argmax(test_y, 1)))
        print("Prediction: ", self.sess.run(tf.argmax(self.scene_model, 1), feed_dict={self.scene_X: test_x, self.scene_keep_prob: 1}))

    def predict(self, image):
        image = cv2.resize(image, (self.width, self.height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(-1, self.width, self.height, 1)

        result = self.sess.run(tf.argmax(self.scene_model, 1), feed_dict={self.scene_X: image, self.scene_keep_prob: 1})
        #result = self.sess.run(self.scene_model, feed_dict={self.scene_X: image, self.scene_keep_prob: 1})
        print(self.kind_scene[result[0]])
        return result[0]
