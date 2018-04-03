import cv2
import os
import csv
from NN.scene_model import Scene_Model
import random
import numpy as np
from PIL import Image
import pytesseract as tes

from pytesseract import pytesseract as pt

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

#make_rec()
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

def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

def extract():
    tes.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
    image = "111.png"
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    thresh = 127
    image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY_INV)[1]

    image = crop_img(image, 0.75)

    result = tes.image_to_string(image)

    print(result)

    cv2.imshow("ads", image)
    cv2.waitKey(0)

def extract2():
    tes.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
    image = "_data/ocr_test/11.png"
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)


    image = cv2.GaussianBlur(image, (3,3),1)
    thresh = 127
    image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY_INV)[1]
    #image = crop_img(image, 0.75)

    print(tes.image_to_string(image, lang='chi_sim'))

    #(w, h) = image.shape
    #M = cv2.getRotationMatrix2D((h / 2, w / 2), 5, 1)
    #image = cv2.warpAffine(image, M, (h, w))

    '''
    y, x = image.shape
    long = 150
    for i in range(0,x, 50):
        for j in range(0, y, 50):
            try:
                im = image[i:j, i+long:j+long]
                print(tes.image_to_string(im, lang='eng', boxes=False, config='--psm 10 --eom 3 -c tessedit_char_whitelist=0123456789'))
            except:
              pass
    '''
    #print(tes.image_to_string(im))

    cv2.imshow("asd", image)
    cv2.waitKey(0)


extract2()