import cv2
import os
import argparse
from NN.Make_data import Make_SceneData
from NN.scene_model import Scene_Model

def train_scene(play):
    s = Scene_Model()
    play = ["180401DUSANKT", "180401KIALG", "180401NCLT", "180401NESAM", "180401SKHW", "180403KIASK", "180403KTNE", "180403LGDUSAN", "180403LTHW"]

    s.load_data(play)

    s.make_model()
    s.train()
    s.test()

def test_scene(image_name, t):
    s = Scene_Model()

    image = cv2.imread(image_name)
    s.make_model()
    if not (t):
        t = 0.7
    print(s.predict(image, int(t))+1)

def make_scene_data(l):
    s = Make_SceneData()

    for p in l:
        file_name = os.path.splitext(os.path.basename(p))[0]
        s.set_path(file_name)
        s.set_video(p)
        s.save_image_with_frame_interval(start=0)

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--videos', nargs='+', help='Input Video name')
    parser.add_argument('-i', '--image', help='Input Image Name in Test Model')

    parser.add_argument('-m', '--make', help='Train Model', action='store_true')
    parser.add_argument('-t', '--train', help='Train Model', action='store_true')
    parser.add_argument('-T', '--test', help='test Model', action='store_true')

    parser.add_argument('--threshold', help='Threshold in Test Model')

    args = parser.parse_args()

    if(args.make):
        make_scene_data(args.video)

    if(args.train):
        train_scene(args.video)

    if(args.test):
        test_scene(args.image, args.threshold)

'''

train_scene(1)
#test_scene()

#make_scene_data()