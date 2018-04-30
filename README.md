# UnderConstruction
# Make Annotation in KBO
Classify the scene in KBO game(video), and make an annotation.

[what is kbo?](https://www.koreabaseball.com "Korea Baseball League")

## Models
![Flow_chart](/PNG/flow_chart.png)

#### Scene Classifier
![Flow_chart](/PNG/scene_classify.png)
- 1 : Pitching and Batting Scene
- 2 : Closeup Batter
- 3 : Closeup Others
- 4 : Closeup Coach
- 5 : Gallery
- 6 ~ 9 : Center, Right Field
- 10 : etc. (Ad., before playing)
- 11 ~ 13 : Left Field

##### Field Classifier
![Flow_chart](/PNG/field_classify.png)
- 6 : About 1st Base
- 7 : About OutField
- 8 : About Right OutField
- 9 : About 2nd Base and InField
- 11 : About 3rd Base (with Zero-Shot Laerning)
- 12 : About Left OutField (with Zero-Shot Laerning)
- 13 : About SS

##### Zero-Shot Learning
![Flow_chart](/PNG/zero_shot.png)

#### Motion Classifier
Underconstruct

#### Web Data
![Flow_chart](/PNG/TextBroadcasting.png)

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

|start | end | label|
|------|-----|------|
|0|35|10|
|36|40|3|
|41|52|1|
|...|...|...|


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

### preparation for Testing Scene Classifier Model:
- Test
  ````
  python train_test.py -T --image path/to/image --threshold 0.7
  ````
  - `-T or --test` Flag, when you want to testing.
  - `--image` Image, which you tend to test.
  - `--threshold` Threshold, when you predict label.


### Training Motion Classifier Model:
### preparation for Testing Motion Classifier Model:


### preparation for Testing(Video):
- Test
  ````
  python main.py -v path/to/video
  ````
  - `-v or --video` input video


