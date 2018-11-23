import numpy as np
import tensorflow as tf
import cv2
import xml.etree.cElementTree as ET


class Detect_Model():
    def __init__(self, sess, istest=0):
        self.sess = sess

        self.count = 0

        self.classes = ["player", "pitcher", "batter", "catcher", "referee"]

        self.width = 416
        self.height = 416
        self.score_threshold = 0.3
        self.iou_threshold = 0.3

        self.X = tf.placeholder(tf.float32, shape=[1, self.height, self.width, 3])

        with tf.variable_scope("object"):
            tmp = self.conv_layer(3, 3, 16, self.X, name="C1")
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
            print(self.model)

        if(istest):
            all_vars = tf.global_variables()
            object = [k for k in all_vars if k.name.startswith("object")]
            saver = tf.train.Saver(object)
            saver.restore(self.sess, "./_model/detect/detect.ckpt")

        else:
            all_vars = tf.global_variables()
            object = [k for k in all_vars if k.name.startswith("object")]
            object = sorted(object, key=lambda x: x.name)
            self.weights_loader("./_model/detect/baseball.weights", "./_model/detect/detect.ckpt", object)

    def predict(self, image):
        h, w, c = image.shape
        ratio_h = h / 416
        ratio_w = w / 416

        image_data = cv2.resize(image, (self.height, self.width), interpolation=cv2.INTER_CUBIC)
        image_data = np.array(image_data, dtype='f')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        #print(self.sess.run(self.tmp, feed_dict={self.X: image_data}))
        predictions = self.sess.run(self.model, feed_dict={self.X: image_data})
        result = self.postprocess(predictions, self.score_threshold, self.iou_threshold)

        if (result):
            result = [
                        [bbox[0][0] * ratio_w, bbox[0][1] * ratio_h, bbox[0][2] * ratio_w, bbox[0][3] * ratio_h, bbox[2]]
                        for bbox in result
                            if bbox[0][0] > 0 and bbox[0][1] > 0 and bbox[0][2] > 0 and bbox[0][3] > 0 #and bbox[2] != "referee"
                      ]

        return result

    def postprocess(self, predictions, score_threshold, iou_threshold):
        # from https://github.com/simo23/tinyYOLOv2
        n_grid_cells = 13
        n_b_boxes = 5
        classes = ["player", "pitcher", "batter", "catcher", "referee"]
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
                        thresholded_predictions.append([[left, top, right, bottom], final_confidence * best_class_score,classes[best_class]])

        thresholded_predictions.sort(key=lambda tup: tup[1], reverse=True)
        nms_predictions = self.non_maximal_suppression(thresholded_predictions, iou_threshold)
        return nms_predictions

    def weights_loader(self, weights_path, ckpt_path, object):
        # from https://github.com/simo23/tinyYOLOv2
        loaded_weights = np.fromfile(weights_path, dtype='f')
        loaded_weights = loaded_weights[4:]

        offset = 0
        biases, kernel_weights, offset = self.load_conv_layer_bn(loaded_weights, [3, 3, 3, 16], offset)
        self.sess.run(tf.assign(object[0], kernel_weights))
        self.sess.run(tf.assign(object[1], biases))

        biases, kernel_weights, offset = self.load_conv_layer_bn(loaded_weights, [3, 3, 16, 32], offset)
        self.sess.run(tf.assign(object[2], kernel_weights))
        self.sess.run(tf.assign(object[3], biases))

        biases, kernel_weights, offset = self.load_conv_layer_bn(loaded_weights, [3, 3, 32, 64], offset)
        self.sess.run(tf.assign(object[4], kernel_weights))
        self.sess.run(tf.assign(object[5], biases))

        biases, kernel_weights, offset = self.load_conv_layer_bn(loaded_weights, [3, 3, 64, 128], offset)
        self.sess.run(tf.assign(object[6], kernel_weights))
        self.sess.run(tf.assign(object[7], biases))

        biases, kernel_weights, offset = self.load_conv_layer_bn(loaded_weights, [3, 3, 128, 256], offset)
        self.sess.run(tf.assign(object[8], kernel_weights))
        self.sess.run(tf.assign(object[9], biases))

        biases, kernel_weights, offset = self.load_conv_layer_bn(loaded_weights, [3, 3, 256, 512], offset)
        self.sess.run(tf.assign(object[10], kernel_weights))
        self.sess.run(tf.assign(object[11], biases))

        biases, kernel_weights, offset = self.load_conv_layer_bn(loaded_weights, [3, 3, 512, 1024], offset)
        self.sess.run(tf.assign(object[12], kernel_weights))
        self.sess.run(tf.assign(object[13], biases))

        biases, kernel_weights, offset = self.load_conv_layer_bn(loaded_weights, [3, 3, 1024, 1024], offset)
        self.sess.run(tf.assign(object[14], kernel_weights))
        self.sess.run(tf.assign(object[15], biases))

        biases, kernel_weights, offset = self.load_conv_layer(loaded_weights, [1, 1, 1024, 50], offset)
        self.sess.run(tf.assign(object[16], kernel_weights))
        self.sess.run(tf.assign(object[17], biases))

        self.saver = tf.train.Saver(object)
        self.saver.save(self.sess, ckpt_path)

    @staticmethod
    def load_conv_layer_bn(loaded_weights, shape, offset):
        #from https://github.com/simo23/tinyYOLOv2
        n_kernel_weights = shape[0] * shape[1] * shape[2] * shape[3]
        n_output_channels = shape[-1]
        n_bn_mean = n_output_channels
        n_bn_var = n_output_channels
        n_biases = n_output_channels
        n_bn_gamma = n_output_channels

        biases = loaded_weights[offset:offset + n_biases]
        offset = offset + n_biases
        gammas = loaded_weights[offset:offset + n_bn_gamma]
        offset = offset + n_bn_gamma
        means = loaded_weights[offset:offset + n_bn_mean]
        offset = offset + n_bn_mean
        var = loaded_weights[offset:offset + n_bn_var]
        offset = offset + n_bn_var
        kernel_weights = loaded_weights[offset:offset + n_kernel_weights]
        offset = offset + n_kernel_weights

        kernel_weights = np.reshape(kernel_weights, (shape[3], shape[2], shape[0], shape[1]), order='C')

        for i in range(n_output_channels):
            scale = gammas[i] / np.sqrt(var[i] + 1e-3)
            kernel_weights[i, :, :, :] = kernel_weights[i, :, :, :] * scale
            biases[i] = biases[i] - means[i] * scale

        kernel_weights = np.transpose(kernel_weights, [2, 3, 1, 0])

        return biases, kernel_weights, offset

    @staticmethod
    def load_conv_layer(loaded_weights, shape, offset):
        # from https://github.com/simo23/tinyYOLOv2
        n_kernel_weights = shape[0] * shape[1] * shape[2] * shape[3]
        n_output_channels = shape[-1]
        n_biases = n_output_channels

        biases = loaded_weights[offset:offset + n_biases]
        offset = offset + n_biases
        kernel_weights = loaded_weights[offset:offset + n_kernel_weights]
        offset = offset + n_kernel_weights

        kernel_weights = np.reshape(kernel_weights, (shape[3], shape[2], shape[0], shape[1]), order='C')
        kernel_weights = np.transpose(kernel_weights, [2, 3, 1, 0])
        return biases, kernel_weights, offset

    @staticmethod
    def conv_layer(filter_size, fin, fout, din, name, activation="leakyrelu"):
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

    @staticmethod
    def pool(din, option, padding, strides=2):
        pool = tf.nn.max_pool(din, ksize=[1, 2, 2, 1], strides=[1, strides, strides, 1], padding=padding)
        return pool

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        out = e_x / e_x.sum()
        return out

    @staticmethod
    def iou(boxA,boxB):
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
            return nms_predictions
        else:
          return None

    def evaluation(self):
        score = {"player":[0.0, 0.0], "pitcher":[0.0, 0.0], "batter":[0.0, 0.0], "catcher":[0.0, 0.0], "referee":[0.0, 0.0]}
        label_xml_path = "_data/_player/label_xml/"
        image_path = "_data/_player/images/"

        filenames = [str(i).zfill(5) for i in range(0, 279)]
        print(filenames)
        image_data = [[cv2.imread(image_path + i+".jpg"), i] for i in filenames]

        for filename in filenames:
            image = cv2.imread(image_path + filename +".jpg")
            xml = open(label_xml_path + filename + ".xml")

            tree = ET.parse(xml)
            root = tree.getroot()

            gt = []
            for obj in root.iter('object'):
                cls = obj.find('name').text
                xmlbox = obj.find('bndbox')
                gt.append([float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text), cls])

            player_gt = [i[:4] for i in gt if i[4] == "player"]
            pitcher_gt = [i[:4] for i in gt if i[4] == "pitcher"]
            batter_gt = [i[:4] for i in gt if i[4] == "batter"]
            catcher_gt = [i[:4] for i in gt if i[4] == "catcher"]
            referee_gt = [i[:4] for i in gt if i[4] == "referee"]


            result = self.predict(image)
            if(result):
                player_pre = [i[:4] for i in result if i[4] == "player"]
                pitcher_pre = [i[:4] for i in result if i[4] == "pitcher"]
                batter_pre = [i[:4] for i in result if i[4] == "batter"]
                catcher_pre = [i[:4] for i in result if i[4] == "catcher"]
                referee_pre = [i[:4] for i in result if i[4] == "referee"]

                score["player"][1] = score["player"][1] + len(player_pre)
                score["pitcher"][1] = score["pitcher"][1] + len(pitcher_pre)
                score["batter"][1] = score["batter"][1] + len(batter_pre)
                score["catcher"][1] = score["catcher"][1] + len(catcher_pre)
                score["referee"][1] = score["referee"][1] + len(referee_pre)

                score["player"][0] = score["player"][0] + self.is_hit(player_pre, player_gt)
                score["pitcher"][0] = score["pitcher"][0] + self.is_hit(pitcher_pre, pitcher_gt)
                score["batter"][0] = score["batter"][0] + self.is_hit(batter_pre, batter_gt)
                score["catcher"][0] = score["catcher"][0] + self.is_hit(catcher_pre, catcher_gt)
                score["referee"][0] = score["referee"][0] + self.is_hit(referee_pre, referee_gt)

        print(score)

        print(score["player"][0] / score["player"][1])
        print(score["pitcher"][0] / score["pitcher"][1])
        print(score["batter"][0] / score["batter"][1])
        print(score["catcher"][0] / score["catcher"][1])
        print(score["referee"][0] / score["referee"][1])

        mAP = (score["player"][0]/score["player"][1] + score["pitcher"][0]/score["pitcher"][1] + score["batter"][0]/score["batter"][1] + score["catcher"][0]/score["catcher"][1] + score["referee"][0]/score["referee"][1])/5
        print(mAP)

    def is_hit(self, pre, gt):
        count = 0
        for p in pre:
            for g in gt:
                if (self.iou(p, g) > 0.5):
                    count = count + 1
                    break

        return count

