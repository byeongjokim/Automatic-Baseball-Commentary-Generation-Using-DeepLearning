import os
import tensorflow as tf

import xml.etree.ElementTree as elemTree
import cv2

from Vision.Model.detect import Detect_Model


def player_detector_eval():
    image_folder = "../_player/images/"
    label_folder = "../_player/label_xml/"

    gt_folder = "../_player/input/ground-truth/"
    dt_folder = "../_player/input/detection-results/"

    test_datset = []

    for i in range(0, 279):
        fname = str(i).zfill(5)
        
        imgfile = image_folder + fname + ".jpg"
        xmlfile = label_folder + fname + ".xml"
        
        img = cv2.imread(imgfile)

        tree = elemTree.parse(xmlfile)
        root = tree.getroot()
        objs = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            objs.append([bndbox.findtext("xmin"), bndbox.findtext("ymin"), bndbox.findtext("xmax"), bndbox.findtext("ymax"), obj.findtext("name")])
        
        test_datset.append({"image":img, "objects":objs, "fname":fname, "predicts":None})


    sess = tf.Session()
    model = Detect_Model(sess=sess, istest=1)

    for i in range(len(test_datset)):
        predicted = model.predict(test_datset[i]["image"])
        with open(dt_folder+"/"+test_datset[i]["fname"]+".txt", "w") as f:
            if(predicted):
                for p in predicted:
                    f.write(str(p[4]) +" "+ str(p[5]) +" "+ str(p[0]) +" "+ str(p[1]) +" "+ str(p[2]) +" "+ str(p[3]) +"\n")
        with open(gt_folder+"/"+test_datset[i]["fname"]+".txt", "w") as f:
            for p in test_datset[i]["objects"]:
                f.write(str(p[4]) +" "+ str(p[0]) +" "+ str(p[1]) +" "+ str(p[2]) +" "+ str(p[3]) +"\n")
        print(predicted)
        print(test_datset[i]["objects"])


player_detector_eval()