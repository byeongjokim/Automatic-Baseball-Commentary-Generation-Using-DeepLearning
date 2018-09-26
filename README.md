# Make Comments in KBO video
In korea, there is a baseball game called [kbo](https://www.koreabaseball.com "Korea Baseball League"), and text broadcasting service [site](https://sports.news.naver.com/kbaseball/schedule/index.nhn). But the text service has no reality. It just send the sentences like "1 strike".
<br>
This is a program which can make comments about the baseball game with a video and web data using deep learning and ontology.


## Models
![Flow_chart](/PNG/flow_chart.png)

When video playing, two thread are runned. One is about scene data and the other is about web data. The thread about web data crawls the text broadcasting, and save it to the ontology. And the other thread about scene data classifies the scene using CNN, classifies the person's motion who locates in the center of video using YOLO, CNN and RNN. With all of data which can retrieve from scene and ontology, the comments are created.


### Scene Classifier
![Scene](/PNG/scene_annotation.png)

- Using Vgg16
- 13 classes (10 classes data)
- training data : highlights videos of baseball game in 15 days


#### Labeling
![Scene](/PNG/scene_labeling.png)
- 1 : Batter box
- 2 : Closeup Batter
- 3 : Closeup Players
- 4 : Closeup Coach
- 5 : Gallery
- 6 ~ 9 : Center, Right Field
- 10 : etc. (Ad., before playing)
- 11 ~ 13 : Left Field


#### About Field
![Scene](/PNG/field_classify.png)

- 6 : About 1st Base
- 7 : About OutField
- 8 : About Right OutField
- 9 : About 2nd Base and InField
- 11 : About 3rd Base (with Zero-Shot Laerning)
- 12 : About Left OutField (with Zero-Shot Laerning)
- 13 : About SS


#### Zero-Shot Learning
![Scene](/PNG/zero_shot.png)

- using 0 3rd Base data, 0 Left OutField data.
- In Baseball Game, There are few data of 3rd Base and Right OutField than others. But I can train these two data with others. Almost Baseball Field have symmetrical characters. So I can get these data with flipping other training data.
- 3rd Base <-> 1st Base
- Left OutField <-> Right OutField


---


### Motion Classifier
![Motion](/PNG/motion_classifier.png)

- using CNN + RNN
	- CNN is the encoder part of auto-encoder model
- make dataset with interval
	- because of other process time, the motion model can get people's images with every some intervals (not real time)
    - 5 ~ 20 frames intervals
    - when training the model, apply intervals in train data


#### Labeling
![Motion](/PNG/motion_labeling.png)

- 0 : Batting Waiting
- 1 : Batting
- 2 : Throwing
- 3 : Pitching
- 4 : Catching - catcher
- 5 : Catching - fielder
- 6 : Running
- 7 : Walking


#### Motion dataset
![Motion](/PNG/motion_dataset.png)

---

### Web Data
![Flow_chart](/PNG/TextBroadcasting.png)

- In [N company](http://www.naver.com), there is a KBO Text Broadcasting site. I can get text data in real time, when game is playing. [Text Broadcasting](http://sports.news.naver.com/kbaseball/schedule/indexnhn)

---

## How to run

### Requirements
- Python3
- Tensorflow
- Opencv3


### Demo:
- Test
  ````
  python main.py -v path/to/video
  ````
  - `-v or --video` input video


