import tensorflow as tf
import numpy as np

class Situation_Model(object):
    def __init__(self, sess):
        self.situations = ["strike", "ball", "foul", "hit", "ground", "flying", "etc"]

        self.max_length = 30
        self.human_max_length = 6

        self.input_length = 256
        self.scene_embeding_length = 64
        self.human_embeding_length = 32

        self.hidden_length = 256

        self.height = 416
        self.width = 416
        self.rgb = 3

        self.featuremap_pad = np.zeros((1, 7, 7, 512), np.float32)

        self.human_height = 50
        self.human_width = 50
        self.human_rgb = 3

        self.human_pad = np.zeros((self.human_height, self.human_width, self.human_rgb), np.float32)
        self.human_pad_max = np.zeros((1, self.human_max_length, self.human_height, self.human_width, self.human_rgb), np.float32)

        self.total_ckpt = "./_model/situation/total.ckpt"

        self.sess = sess
        self.batch_size = 1

        self.X = tf.placeholder(tf.float32, [self.batch_size, self.max_length, 7, 7, 512])
        self.H = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.human_max_length, self.human_height, self.human_width, self.human_rgb])
        self.L = tf.placeholder(tf.int32, [self.batch_size])

        self.model(self.H, self.L, self.X)

        all_vars = tf.global_variables()
        situ = [k for k in all_vars if k.name.startswith("situation")]
        saver = tf.train.Saver(situ)
        saver.restore(self.sess, self.total_ckpt)

    def model(self, H, L, scene_model, Y=None):
        with tf.variable_scope("situation"):
            scene_reshape = tf.reshape(scene_model, [-1, 7 * 7 * 512])

            scene_embed_W1 = tf.get_variable("scene_embed_W1", shape=[7 * 7 * 512, 128], initializer=tf.contrib.layers.xavier_initializer())
            scene_embed_b1 = tf.get_variable("scene_embed_b1", shape=[128], initializer=tf.contrib.layers.xavier_initializer())
            scene_embed_fc1 = tf.nn.relu(tf.matmul(scene_reshape, scene_embed_W1) + scene_embed_b1)

            scene_embed_W2 = tf.get_variable("scene_embed_W2", shape=[128, self.scene_embeding_length], initializer=tf.contrib.layers.xavier_initializer())
            scene_embed_b2 = tf.get_variable("scene_embed_b2", shape=[self.scene_embeding_length], initializer=tf.contrib.layers.xavier_initializer())
            scene_embed_fc2 = tf.nn.relu(tf.matmul(scene_embed_fc1, scene_embed_W2) + scene_embed_b2)

            scene_embed = tf.reshape(scene_embed_fc2, [self.batch_size, self.max_length, self.scene_embeding_length])

            h = tf.reshape(H, [self.batch_size * self.max_length * self.human_max_length, self.human_height, self.human_width, self.human_rgb])

            human_C1 = self.conv_layer(filter_size=3, fin=self.human_rgb, fout=64, din=h, name="human_C1")
            human_P1 = self.pool(human_C1, option="maxpool")

            human_C2 = self.conv_layer(filter_size=3, fin=64, fout=64, din=human_P1, name="human_C2")
            human_C2_2 = self.conv_layer(filter_size=3, fin=64, fout=64, din=human_C2, name="human_C2_2")
            human_P2 = self.pool(human_C2_2, option="maxpool")

            human_C3 = self.conv_layer(filter_size=3, fin=64, fout=64, din=human_P2, name="human_C3")
            human_C3_2 = self.conv_layer(filter_size=3, fin=64, fout=128, din=human_C3, name="human_C3_2")
            human_P3 = self.pool(human_C3_2, option="maxpool")

            human_C4 = self.conv_layer(filter_size=3, fin=128, fout=128, din=human_P3, name="human_C4")
            human_C4_2 = self.conv_layer(filter_size=3, fin=128, fout=512, din=human_C4, name="human_C4_2")
            human_P4 = self.pool(human_C4_2, option="maxpool")

            human_reshape = tf.reshape(human_P4, [-1, 4 * 4 * 512])

            human_embed_W1 = tf.get_variable("human_embed_W1", shape=[4 * 4 * 512, 128], initializer=tf.contrib.layers.xavier_initializer())
            human_embed_b1 = tf.get_variable("human_embed_b1", shape=[128], initializer=tf.contrib.layers.xavier_initializer())
            human_embed_fc1 = tf.nn.relu(tf.matmul(human_reshape, human_embed_W1) + human_embed_b1)

            human_embed_W2 = tf.get_variable("human_embed_W2", shape=[128, self.human_embeding_length], initializer=tf.contrib.layers.xavier_initializer())
            human_embed_b2 = tf.get_variable("human_embed_b2", shape=[self.human_embeding_length], initializer=tf.contrib.layers.xavier_initializer())
            human_embed_fc2 = tf.nn.relu(tf.matmul(human_embed_fc1, human_embed_W2) + human_embed_b2)

            human_embed = tf.reshape(human_embed_fc2, [self.batch_size, self.max_length, self.human_max_length * self.human_embeding_length])

            rnn_input = tf.concat([scene_embed, human_embed], 2)

            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_length)
            all_outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_input, sequence_length=L, dtype=tf.float32)
            self.outputs = self.last_relevant(all_outputs, L)

            last_W = tf.get_variable("last_W", shape=[self.hidden_length, len(self.situations)], initializer=tf.contrib.layers.xavier_initializer())
            last_b = tf.Variable(tf.constant(0.1, shape=[len(self.situations)]), name="last_b")

            self.logits = tf.nn.xw_plus_b(self.outputs, last_W, last_b, name="logits")
            self.softmax_logits = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

    def predict(self, scene_queue, human_queue, L):
        while (len(human_queue[-1]) < self.human_max_length):
            human_queue[-1].append(self.human_pad)

        X, H, L = self._fill_pad(scene_queue[-1 * self.max_length : ], human_queue[-1 * self.max_length : ], L)

        predictions, softmax_logits = self.sess.run([self.predictions, self.softmax_logits],
                                                                    feed_dict={self.X: [X],
                                                                               self.H: [H],
                                                                               self.L: [L]})

        return scene_queue, human_queue, predictions[0], max(softmax_logits)

    def _fill_pad(self, scene_queue, human_queue, L):
        if (L < self.max_length):
            while(len(scene_queue) < self.max_length):
                scene_queue = np.concatenate((scene_queue, self.featuremap_pad))
            while(len(human_queue) < self.max_length):
                human_queue = np.concatenate((human_queue, self.human_pad_max))
            return scene_queue, human_queue, L
        else:
            return scene_queue, human_queue, self.max_length

    def last_relevant(self, seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)

    def conv_layer(self, filter_size, fin, fout, din, name):
        with tf.variable_scope(name):
            W = tf.get_variable(name=name + "_W", shape=[filter_size, filter_size, fin, fout],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name=name + "_b", shape=[fout], initializer=tf.contrib.layers.xavier_initializer(0.0))
            C = tf.nn.conv2d(din, W, strides=[1, 1, 1, 1], padding='SAME')
            R = tf.nn.relu(tf.nn.bias_add(C, b))
            return R

    def pool(self, din, option='maxpool'):
        if (option == 'maxpool'):
            pool = tf.nn.max_pool(din, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        elif (option == 'avrpool'):
            pool = tf.nn.avg_pool(din, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        else:
            return din
        return pool

