import cv2
import os
import csv
from NN.scene_model import Scene_Model
import random
import numpy as np
from PIL import Image
import pytesseract as tes

def get_closeup():
    s = Scene_Model()
    s.make_model()
    play = ["20171028KIADUSAN", "20171029KIADUSAN", "20171030KIADUSAN"]
    # play = "20171030KIADUSAN"
    for p in play:
        folder_path = "./_data/" + p + "/"
        image = [str(i) + ".jpg" for i in range(0, 14000)]
        f = open(p + ".txt", "w")

        for i in image:
            im = cv2.imread(folder_path + i)
            if (s.predict(im) == 3):  # closeup
                f.write(i + "\n")
        f.close()

def get_data(play):
    folder_path = "./_data/" + play + "/"
    txt_path = folder_path + play+"_closeup.txt"
    f = open(txt_path, 'r')
    data =[]
    while True:
        line = f.readline()
        if not line:
            break
        data.append(line[:-1])
    return data

def make_rec():
    #play = ["20171028KIADUSAN", "20171029KIADUSAN", "20171030KIADUSAN"]
    play = "20171029KIADUSAN"
    image = []

    data = get_data(play)
    folder_path = "./_data/" + play + "/"

    for i in data:
        image_path = folder_path + i
        im = cv2.imread(image_path)

        f = open(play + "_backnumber.csv", 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)

        print(i)
        r = cv2.selectROI(im)
        # ((r[1], r[0]) ~ (r[1]+r[3], r[0]+r[2]))
        print(r[0], r[1], r[2], r[3])
        if not (r[2] == 0 or r[3] == 0):
            if not (r[0] == 0 and r[2] == 0 or r[1] == 0 and r[3] == 0):
                wr.writerow([i, r[0], r[1], r[2], r[3]])
        f.close()

make_rec()
def check():
    play = ["20171029KIADUSAN", "20171030KIADUSAN"]

    p = "20171029KIADUSAN"
    folder_path = "./_data/" + p + "/"
    csv_path = folder_path + p + "_backnumber.csv"

    f = open(csv_path, "r")
    reader = csv.reader(f)
    data = []
    for line in reader:
        data.append({"image":line[0], "start_y":int(line[1]), "start_x":int(line[2]), "width": int(line[3]), "height": int(line[4])})

    test_image = data[random.randrange(0,100)]

    im = cv2.imread(folder_path + test_image["image"])
    img = cv2.rectangle(im, (test_image["start_y"], test_image["start_x"]), (test_image["start_y"]+test_image["width"], test_image["start_x"]+test_image["height"]), (0,255,0), 3)

    cv2.imshow("asd", img)
    cv2.waitKey(0)

#check()
#make_rec()

def extract():
    tes.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
    p = "20171030KIADUSAN"
    folder_path = "./_data/" + p + "/"
    image = folder_path + "12205.jpg"
    test_image = "../../123.png"
    print(image)
    #results = tes.image_to_string(Image.open(test_image))
    results = tes.image_to_string(cv2.imread(test_image))
    print(results)
    cv2.imshow("ads", cv2.imread(test_image))
    cv2.waitKey(0)



#extract()