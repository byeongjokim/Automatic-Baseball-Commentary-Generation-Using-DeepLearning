# UnderConstruction
# Make Annotation in KBO
Classify the scene in KBO game(video), and make an annotation.

[what is kbo?](https://www.koreabaseball.com "Korea Baseball League")

<br>

## Models
![Flow_chart](/PNG/flow_chart.png)

explain about Models

<br>

### Scene Classifier
![Flow_chart](/PNG/scene_classify.png)

explain about scene classifier

<br>

#### Labeling
![Flow_chart](/PNG/labeling.png)
- 1 : Pitching and Batting Scene
- 2 : Closeup Batter
- 3 : Closeup Others
- 4 : Closeup Coach
- 5 : Gallery
- 6 ~ 9 : Center, Right Field
- 10 : etc. (Ad., before playing)
- 11 ~ 13 : Left Field

<br>

#### About Field
![Flow_chart](/PNG/field_classify.png)
- 6 : About 1st Base
- 7 : About OutField
- 8 : About Right OutField
- 9 : About 2nd Base and InField
- 11 : About 3rd Base (with Zero-Shot Laerning)
- 12 : About Left OutField (with Zero-Shot Laerning)
- 13 : About SS

<br>

#### Zero-Shot Learning
![Flow_chart](/PNG/zero_shot.png)
- using 0 3rd Base data, 0 Left OutField data.
- In Baseball Game, There are few data of 3rd Base and Right OutField than others. But I can train these two data with others. Almost Baseball Field have symmetrical characters. So I can get these data with flipping other training data.
- 3rd Base <-> 1st Base
- Left OutField <-> Right OutField

<br>

---

### Motion Classifier
Underconstruct

---

### Web Data
![Flow_chart](/PNG/TextBroadcasting.png)
- In [N company](http://www.naver.com), there is a KBO Text Broadcasting site. I can get text data in real time, when game is playing. [Text Broadcasting](http://sports.news.naver.com/kbaseball/schedule/indexnhn)

---

## Train and Test

### Requirements
- Python3
- Tensorflow
- Opencv3

### Training Scene Classifier Model:
- Make DataSet
  ````
    python train_test.py -m --videos path/to/video path/to/another/video
    ````
    - `-m or --make` Flag, when you want to make train_data images.
    - `--videos` Videos, which you tend to make dataset.
    - you can find images in **_data/video_name** folder.
  - make **video_name.csv** in **_data/video_name**.
    - in **video_name.csv** you should write start number, end number and label of **_data/video_name** images. [View Scene Label](/PNG/field_classify.png)

![Flow_chart](/PNG/scene_labeling.png)


- Train
  ````
  python train_test.py -t --videos _data/video_name _data/video_name
  ````
  - `-t or --train` Flag, when you want to training.
  - `--videos` Videos, which you tend to train.


- **In my project**
  - Using n KBO HighLight Videos(2018. 04.01 ~ 2018.04.30) in [YouTube](https://www.youtube.com/playlist?list=PL7MQjbfOyOE19FCi85BcECO-zNYQcDbE0)
    - Download DataSet in [here](https://github.com/byeongjokim/KBO_annotation) to _data/**
    - Donwload Trained Model (tensorflow) in [here](https://github.com/byeongjokim/KBO_annotation) to _model/**

### Testing Scene Classifier Model:
- Test
  ````
  python train_test.py -T --image path/to/image --threshold 0.7
  ````
  - `-T or --test` Flag, when you want to testing.
  - `--image` Image, which you tend to test.
  - `--threshold` Threshold, when you predict label.

---

### Training Motion Classifier Model:
### Testing Motion Classifier Model:

---

### Demo:
- Test
  ````
  python main.py -v path/to/video
  ````
  - `-v or --video` input video


