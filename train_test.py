import cv2
import os
import argparse
#from NN.Make_data import Make_SceneData
from NN.scene_model import Scene_Model

def train_scene(play):
    s = Scene_Model()
    play = ["180401HTLG", "180401NCLT", "180401OBKT", "180401SKHH", "180401WOSS",
            "180403HTSK", "180403KTWO", "180403LGOB", "180403LTHH", "180403SSNC",
            "180404HTSK", "180404KTWO", "180404LGOB", "180404LTHH", "180404SSNC",
            "180405KTWO", "180405SSNC",
            "180406LGLT", "180406WOHT",
            "180407HHKT", "180407LGLT", "180407NCOB", "180407SSSK", "180407WOHT",
            "180408HHKT", "180408LGLT", "180408NCOB", "180408SSSK", "180408WOHT",
            "180410HTHH", "180410KTNC", "180410OBSS", "180410SKLG", "180410WOLT",
            "180411HTHH", "180411KTNC", "180411OBSS", "180411SKLG", "180411WOLT",
            "180412HTHH", "180412KTNC", "180412OBSS", "180412SKLG", "180412WOLT",
            "180413KTLG", "180413LTHT", "180413NCSK", "180413OBWO", "180413SSHH"
            ]

    s.load_data(play)

    s.make_model()
    s.train()
    s.test()

def test_scene(image_name, t):
    s = Scene_Model()

    image = cv2.imread(image_name)
    s.make_model()
    print(s.predict(image))

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