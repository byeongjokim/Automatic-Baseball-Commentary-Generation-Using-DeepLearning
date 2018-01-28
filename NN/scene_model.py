import cv2
import csv
import tensorflow as tf
import numpy as np
from skimage.measure import compare_ssim as ssim
import os
import operator
import random
from operator import itemgetter
from collections import Counter
from NN.cnn import conv_layer, pool

class Scene_Model():
    num_label = 0
    chk_scene = './_model/scene/scene.ckpt'
    ckpt = tf.train.get_checkpoint_state(("./_model/scene"))

    batch_size = 30
    epoch = 100

    width = 224
    height = 224

    kind_scene = ['field', 'pitcher', 'gallery', 'batter', 'pitchingbatting', 'beforestart', '1', 'coach', 'closeup', '3']

    def __init__(self):
        print("init scene_model")

    def load_data(self):
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
        self.num_label = len(set(_y))

    #vggnet - A
    def make_model(self):
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

        fc0 = tf.reshape(P4, [-1, 7 * 7 * 512])

        with tf.device("/cpu:0"):
            W1 = tf.get_variable("scene_W1", shape=[7 * 7 * 512, 4096],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([4096]))
            fc1 = tf.nn.relu(tf.matmul(fc0, W1) + b1)

            W2 = tf.get_variable("scene_W2", shape=[4096, 1000], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([1000]))
            fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)

            W3 = tf.get_variable("scene_W3", shape=[1000, self.num_label], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([self.num_label]))
            self.scene_model = tf.matmul(fc2, W3) + b3

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

            if (total_cost / total_batch < 0.01):
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
        print("Prediction: ", self.sess.run(tf.argmax(self.scene_model, 1), feed_dict={self.X: test_x, self.scene_keep_prob: 1}))

    def predict(self):
        return 1



class Make_SceneData():
    def __init__(self, path, shape=(320,180),fps=29.970):
        print("sceneData")
        self.path = path
        self.fps = fps
        self.get_data_csv()

        self.width = shape[0]
        self.height = shape[1]

    def get_data_csv(self):
        f = open(self.path, 'r', encoding='utf-8')
        reader = csv.reader(f)
        result = []
        count = 1
        for line in reader:
            if not line:
                pass

            elif(line[0] == str(count)):
                result.append({"SceneNumber": count, "start": int(line[1]), "end": int(float(line[4]) * self.fps) + int(line[1]), "label": None})
                count = count + 1

        self.data = result[:400]
        f.close()

    def save_image_data(self):
        video = cv2.VideoCapture("./_data/20171030KIADUSAN.mp4")

        count = 0
        for i in self.result:
            no_frame = (i["start"] + i["end"]) / 2
            video.set(1, no_frame)
            success, frame = video.read()

            if not success:
                break

            cv2.imwrite("./scene_data/"+i["label"]+"_"+str(count)+".jpg", frame)
            count = count + 1

        return 1



    def mse(self, A, B):
        err = np.sum((A.astype('float') - B.astype('float')) ** 2)
        err /= float(A.shape[0] * A.shape[1])
        return err

    def compare_images(self, A, B):
        m = self.mse(A, B)
        s = ssim(A, B, multichannel=True)
        #print("MSE: %.2f, struct_SSIM: %.2f" % (m, s))
        return s


    def clustering(self):
        train_data = []
        test_data = []

        path = "./scene_data/train/"
        image = []
        for (p, dir, files) in os.walk(path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.jpg':
                    image.append(filename)

        train_data=[]
        for i in image:
            train_data.append({"image":
                cv2.cvtColor(
                    cv2.imread(path + i),
                    cv2.COLOR_BGR2GRAY),
                "label": i.split(".")[0].split("_")[0]})


        video = cv2.VideoCapture("./_data/20171030KIADUSAN.mp4")

        for i in self.data:
            no_frame = (i["start"] + i["end"]) / 2
            video.set(1, no_frame)
            success, frame = video.read()

            if not success:
                break

            test_data.append({"image":frame, "label":None})

        print("made %d test, %d train data" %(len(test_data), len(train_data)))
        print("will calculate simm")

        count = 0
        for i in test_data:
            print(count)

            result = []
            for j in train_data:
                s = self.compare_images(j["image"], cv2.cvtColor(i["image"], cv2.COLOR_BGR2GRAY))

                result.append({"label":j["label"], "ssim":s})

            result.sort(key=operator.itemgetter('ssim'), reverse=True)
            print(result)
            result = result[:3]
            print(result)
            l = [i["label"] for i in result]

            first = result[0]["label"]

            counter = Counter(l)
            print(counter)
            if(counter.most_common()[0][1] == 1):
                label = first
            else:
                label = counter.most_common()[0][0]

            print(label)
            i["label"] = label
            cv2.imwrite("./scene_data/test/"+str(i["label"])+"_"+str(count)+".jpg", i["image"])
            count = count + 1
        #self.result = test_data + train_data


