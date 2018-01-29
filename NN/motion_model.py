import tensorflow as tf
import cv2
from imutils.object_detection import non_max_suppression
from NN.cnn import conv_layer, pool
import os
import numpy as np
import random

chk = './_model/action_full/action.ckpt'
#chk = './_model/action_upper/action.ckpt'
ckpt = tf.train.get_checkpoint_state('./_model/action_full')
#ckpt = tf.train.get_checkpoint_state('./_model/action_upper')
body_cascade = cv2.CascadeClassifier('./_data/cascades/haarcascade_fullbody.xml')
#body_cascade = cv2.CascadeClassifier('./_data/cascades/haarcascade_upperbody.xml')

def get_human(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    people = body_cascade.detectMultiScale(gray, 1.05, 3, flags=cv2.CASCADE_SCALE_IMAGE)
    people = non_max_suppression(people, probs=None, overlapThresh=0.75)

    return people


form = ["walking", "jogging", "running", "pitching"]
#"jogging",


class Data():
    def __init__(self, kth=1, ucf=0, size=[60, 80]):
        if(kth==1):
            self.kth = KTH("./_data/kth/", size=size)
            self.kth_data = self.kth.prepare()
            print(len(self.kth_data), " kth data")
        else:
            self.kth_data=[]

        if(ucf==1):
            self.ucf = UCF("./_data/ucf/", size=size)
            self.ucf_data = self.ucf.prepare()
            print(len(self.ucf_data), " ucf data")
        else:
            self.ucf_data = []


    def make_train_data(self):
        data = self.kth_data + self.ucf_data

        random.shuffle(data)

        x = np.array([d["image"] for d in data])
        _y = np.array([d["label"] for d in data])

        y = np.zeros((len(_y), len(form)))
        y[np.arange(len(_y)), [form.index(i) for i in _y]] = 1

        l = int(len(data) / 10)

        train_x = x[l:]
        train_y = y[l:]
        test_x = x[:l]
        test_y = y[:l]

        return train_x, train_y, test_x, test_y, len(form)

class UCF():
    def __init__(self, path, size):
        self.width = size[0]
        self.height = size[1]
        self.path = path
        self.category = ["pitching", "running"]

        if not(os.path.exists(self.path)):
            print("path error")

    def prepare(self):
        file_list = []
        for c in self.category:
            folder_name = self.path + c
            for f in os.listdir(folder_name):
                file_list.append( {"label" : c, "fileName" : self.path + c+ '/' + f} )

        result = []
        for c in file_list:
            if "avi" in c["fileName"]:
                result = result + self.extract_avi(c)
            elif "jpg" in c["fileName"]:
                result = result + self.extract_photo(c)
            else:
                pass

        return result

    def extract_avi(self, data):
        label = data["label"]
        video = data["fileName"]
        vidcap = cv2.VideoCapture(video)

        return_data = []
        count = 0
        while True:
            count = count + 1
            success, frame = vidcap.read()
            if not(success):
                break
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resize = cv2.resize(frame, (self.width, self.height))
            img = np.array(resize)

            return_data.append({"image": img, "label": label})

        return return_data

    def extract_photo(self, data):
        label = data["label"]
        photo = data["fileName"]

        image = cv2.imread(photo)
        resize = cv2.resize(image, (self.width, self.height))
        img = np.array(resize)

        return_data = [{"image": img, "label": label}]

        return return_data


class KTH():
    def __init__(self, path, size):
        self.width = size[0]
        self.height = size[1]
        self.path = path

        if not(os.path.exists(self.path)):
            print("path error")

    def prepare(self):
        file_name = self.path + 'label.txt'

        lines = [line.rstrip('\n').rstrip('\r').split("\t") for line in open(file_name)]
        lines = [line for line in lines if("run" in line[0] or "jog" in line[0] or "walk" in line[0])]

        #lines = lines[:20]

        label = [l[0].rstrip() for l in lines]
        _frame = [l[-1].split(", ") for l in lines]

        result = []

        for frames, l in zip(_frame, label):
            for frame in frames:
                start = frame.split("-")[0]
                end = frame.split("-")[1]
                form = {"fileName": l, "start": start, "end": end}
                result.append(form)

        data = []
        count = 1

        for d in result:
            data = data + self.extract(d)
            count = count + 1

        return data

    def extract(self, data):

        lab = data["fileName"].split("_")[1]

        video_name = self.path + lab + "/" + data['fileName'] + "_uncomp.avi"

        vidcap = cv2.VideoCapture(video_name)

        count = 0
        return_data = []

        while True:
            success, frame= vidcap.read()
            if not (success):
                break

            if (count > int(data["start"]) and count < int(data["end"]) - 10 and count % 10 == 0):

                people = get_human(frame)

                c = 0
                for (x,y,w,h) in people:
                    person = frame[y:y+h, x:x+w]
                    gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
                    resize = cv2.resize(gray, (self.width, self.height))
                    img = np.array(resize)
                    #cv2.imwrite(str(count)+str(c)+".png", img)
                    return_data.append({"image":img, "label":lab})
                    c = c+1

            elif(count > int(data["end"])-10):
                break

            count = count + 1

        return return_data

class Action():

    batch_size = 30
    epoch = 100

    def __init__(self, num_out, size=[60, 80]):
        print("init action class")

        self.num_out = num_out

        self.width = size[0]
        self.height = size[1]


    def make_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.width, self.height, 1])
        self.Y = tf.placeholder(tf.float32, [None, self.num_out])
        self.keep_prob = tf.placeholder(tf.float32)

        C1_1 = conv_layer(filter_size=3, fin=1, fout=3, din=self.X, name='C1_1')
        C1_2 = conv_layer(filter_size=3, fin=3, fout=9, din=C1_1, name='C1_2')
        P1 = pool(C1_2, option="maxpool")
        P1 = tf.nn.dropout(P1, keep_prob=self.keep_prob)

        C2_1 = conv_layer(filter_size=3, fin=9, fout=27, din=P1, name='C2_1')
        C2_2 = conv_layer(filter_size=3, fin=27, fout=54, din=C2_1, name='C2_2')
        P2 = pool(C2_2, option="maxpool")
        P2 = tf.nn.dropout(P2, keep_prob=self.keep_prob)

        C3_1 = conv_layer(filter_size=3, fin=54, fout=54, din=P2, name='C3_1')
        C3_2 = conv_layer(filter_size=3, fin=54, fout=54, din=C3_1, name='C3_2')
        P3 = pool(C3_2, option="maxpool")
        P3 = tf.nn.dropout(P3, keep_prob=self.keep_prob)

        C4_1 = conv_layer(filter_size=3, fin=54, fout=54, din=P3, name='C4_1')
        C4_2 = conv_layer(filter_size=3, fin=54, fout=54, din=C4_1, name='C4_2')
        P4 = pool(C3_2, option="maxpool")
        P4 = tf.nn.dropout(P4, keep_prob=self.keep_prob)

        print(P4)

        fc0 = tf.reshape(P4, [-1, 8 * 10 * 54])

        with tf.device("/cpu:0"):
            W1 = tf.get_variable("W1", shape=[8 * 10 * 54, 4096], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([4096]))
            fc1 = tf.nn.relu(tf.matmul(fc0, W1) + b1)

            W2 = tf.get_variable("W2", shape=[4096, 4096], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([4096]))
            fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)

            W3 = tf.get_variable("W3", shape=[4096, self.num_out], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([self.num_out]))
            self.model = tf.matmul(fc2, W3) + b3

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        # ckpt = tf.train.get_checkpoint_state('./_models')


        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("restore the sess!!")
            self.saver.restore(self.sess, chk)
        else:
            self.sess.run(tf.global_variables_initializer())

    def train(self, train_x, train_y):
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
                batch_y = batch_y.reshape(-1, self.num_out)

                _, cost_val = self.sess.run([self.optimizer, self.cost],
                                            feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.8})

                total_cost = total_cost + cost_val

            if(total_cost/total_batch < 0.01):
                break
            print('Epoch:', '%d' % (e + 1), 'Average cost =', '{:.3f}'.format(total_cost / total_batch))

        print("complete")
        self.saver.save(self.sess, chk)

    def test(self, test_x, test_y):
        is_correct = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        test_x = test_x.reshape(-1, self.width, self.height, 1)
        print('accuracy: ', self.sess.run(accuracy, feed_dict={self.X: test_x, self.Y: test_y, self.keep_prob: 1}) * 100)

        print("Label: ", self.sess.run(tf.argmax(test_y, 1)))
        print("Prediction: ", self.sess.run(tf.argmax(self.model, 1), feed_dict={self.X: test_x, self.keep_prob: 1}))

    def predict(self, data):
        data = data.reshape(-1, self.width, self.height, 1)

        result = self.sess.run(tf.argmax(self.model, 1), feed_dict={self.X: data, self.keep_prob: 1})



        if(result == 0):
            print("person is walking")
            return "person is walking"
        elif(result == 1):
            print("person is jogging")
            return "person is jogging"
        elif(result == 2):
            print("person is runnging")
            return "person is running"
        elif (result == 3):
            print("person is pitching")
            return "person is pitching"
        else:
            return "nothing"

class Action_model():
    def __init__(self, full):
        return 1