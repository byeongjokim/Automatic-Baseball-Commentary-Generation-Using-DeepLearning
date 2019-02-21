import numpy as np
import tensorflow as tf
import cv2

class Detect_Model():
    def __init__(self, sess, istest=0):
        self.sess = sess

        self.classes = ["player", "pitcher", "batter", "catcher", "referee"]

        self.width = 416
        self.height = 416
        self.score_threshold = 0.3
        self.iou_threshold = 0.3

        self.detect_X = tf.placeholder(tf.float32, shape=[1, self.height, self.width, 3])

        with tf.variable_scope("object"):
            tmp = self.conv_layer(3, 3, 16, self.detect_X, name="C1")
            tmp = self.pool(tmp, option="maxpool", padding="VALID")
            tmp = self.conv_layer(3, 16, 32, tmp, name="C2")
            tmp = self.pool(tmp, option="maxpool", padding="VALID")
            tmp = self.conv_layer(3, 32, 64, tmp, name="C3")
            tmp = self.pool(tmp, option="maxpool", padding="VALID")
            tmp = self.conv_layer(3, 64, 128, tmp, name="C4")
            tmp = self.pool(tmp, option="maxpool", padding="VALID")
            tmp = self.conv_layer(3, 128, 256, tmp, name="C5")
            tmp = self.pool(tmp, option="maxpool", padding="VALID")
            tmp = self.conv_layer(3, 256, 512, tmp, name="C6")
            tmp = self.pool(tmp, option="maxpool", padding="SAME", strides=1)
            tmp = self.conv_layer(3, 512, 1024, tmp, name="C7")
            tmp = self.conv_layer(3, 1024, 1024, tmp, name="C8")
            self.model = self.conv_layer(1, 1024, 50, tmp, name="C9", activation="None")

        if(istest):
            all_vars = tf.global_variables()
            object = [k for k in all_vars if k.name.startswith("object")]
            saver = tf.train.Saver(object)
            saver.restore(self.sess, "./_model/detect/detect.ckpt")

    def get_model(self):
        return self.model, self.detect_X

    def predict(self, image):
        h, w, c = image.shape
        ratio_h = h / 416
        ratio_w = w / 416

        image_data = cv2.resize(image, (self.height, self.width), interpolation=cv2.INTER_CUBIC)
        image_data = np.array(image_data, dtype='f')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        predictions = self.sess.run(self.model, feed_dict={self.detect_X: image_data})
        result = self.postprocess(predictions, self.score_threshold, self.iou_threshold)

        if(result):
            result = [
                [int(bbox[0][0] * ratio_w), int(bbox[0][1] * ratio_h), int(bbox[0][2] * ratio_w), int(bbox[0][3] * ratio_h), bbox[2]]
                for bbox in result
                    if bbox[0][0] > 0 and bbox[0][1] > 0 and bbox[0][2] > 0 and bbox[0][3] > 0
            ]

        return result

    def postprocess(self, predictions, score_threshold, iou_threshold):
        # from https://github.com/simo23/tinyYOLOv2
        n_grid_cells = 13
        n_b_boxes = 5
        anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
        thresholded_predictions = []
        predictions = np.reshape(predictions, (13, 13, 5, 10))

        for row in range(n_grid_cells):
            for col in range(n_grid_cells):
                for b in range(n_b_boxes):
                    tx, ty, tw, th, tc = predictions[row, col, b, :5]
                    center_x = (float(col) + self.sigmoid(tx)) * 32.0
                    center_y = (float(row) + self.sigmoid(ty)) * 32.0

                    roi_w = np.exp(tw) * anchors[2 * b + 0] * 32.0
                    roi_h = np.exp(th) * anchors[2 * b + 1] * 32.0

                    final_confidence = self.sigmoid(tc)
                    class_predictions = predictions[row, col, b, 5:]
                    class_predictions = self.softmax(class_predictions)
                    class_predictions = tuple(class_predictions)
                    best_class = class_predictions.index(max(class_predictions))
                    best_class_score = class_predictions[best_class]

                    # Compute the final coordinates on both axes
                    left = int(center_x - (roi_w / 2.))
                    right = int(center_x + (roi_w / 2.))
                    top = int(center_y - (roi_h / 2.))
                    bottom = int(center_y + (roi_h / 2.))

                    if( (final_confidence * best_class_score) > score_threshold):
                        thresholded_predictions.append([[left, top, right, bottom], final_confidence * best_class_score, self.classes[best_class]])

        thresholded_predictions.sort(key=lambda tup: tup[1], reverse=True)
        nms_predictions = self.non_maximal_suppression(thresholded_predictions, iou_threshold)

        return nms_predictions

    def conv_layer(self, filter_size, fin, fout, din, name, activation="leakyrelu"):
        W = tf.get_variable(name=name+"_W", shape=[filter_size, filter_size, fin, fout],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name=name+"_b", shape=[fout],
                            initializer=tf.contrib.layers.xavier_initializer(0.0))
        C = tf.nn.conv2d(din, W, strides=[1, 1, 1, 1], padding='SAME')
        if(activation=="leakyrelu"):
            R = tf.nn.leaky_relu(tf.nn.bias_add(C, b), alpha=0.1)
            return R
        else:
            return C+b

    def pool(self, din, option, padding, strides=2):
        pool = tf.nn.max_pool(din, ksize=[1, 2, 2, 1], strides=[1, strides, strides, 1], padding=padding)
        return pool

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        out = e_x / e_x.sum()
        return out

    def iou(self, boxA,boxB):
        # from https://github.com/simo23/tinyYOLOv2
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        intersection_area = (xB - xA + 1) * (yB - yA + 1)

        boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        if (boxA_area + boxB_area - intersection_area == 0):
            return 100
        else:
            iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
            return iou

    def non_maximal_suppression(self, thresholded_predictions,iou_threshold):
        # from https://github.com/simo23/tinyYOLOv2
        nms_predictions = []

        if(thresholded_predictions):
            nms_predictions.append(thresholded_predictions[0])

            i = 1
            while i < len(thresholded_predictions):
                n_boxes_to_check = len(nms_predictions)

                to_delete = False

                j = 0
                while j < n_boxes_to_check:
                    curr_iou = self.iou(thresholded_predictions[i][0],nms_predictions[j][0])
                    if(curr_iou > iou_threshold ):
                        to_delete = True
                    j = j+1

                if to_delete == False:
                    nms_predictions.append(thresholded_predictions[i])
                i = i+1
            nms_predictions.sort(key=lambda x: x[0][0])
            return nms_predictions
        else:
            return None
