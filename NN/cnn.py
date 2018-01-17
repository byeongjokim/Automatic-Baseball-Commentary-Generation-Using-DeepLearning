import tensorflow as tf

def conv_layer(filter_size, fin, fout, din, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name=name + "_W", shape=[filter_size, filter_size, fin, fout],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name=name + "_b", shape=[fout], initializer=tf.contrib.layers.xavier_initializer(0.0))
        C = tf.nn.conv2d(din, W, strides=[1, 1, 1, 1], padding='SAME')
        R = tf.nn.relu(tf.nn.bias_add(C, b))
        return R


def pool(din, option='maxpool'):
    if (option == 'maxpool'):
        pool = tf.nn.max_pool(din, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    elif (option == 'avrpool'):
        pool = tf.nn.avg_pool(din, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    else:
        return din
    return pool