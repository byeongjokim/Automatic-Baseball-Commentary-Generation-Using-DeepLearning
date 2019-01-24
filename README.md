# Baseball Game Casting with Deep Learning
There have been many studies to make captions or explanations about images and video with deep learning. A combination of CNN and RNN is mainly used to generate captions from the features of image and frame. Recently, the attention model has been combined to explain a more detailed and accurate situation. Through these studies, deep learning can describe the real state of an image Human-Like. However, when casting a sports event, castings are created not only from contextual information about who does what, but also from understanding and reasoning using the player’s information and past knowledge.<br>
This research describes the three models (scene classifier, player detection, motion recognition) to obtain contextual information from the sports frame and the ontology to inference knowledge from past data. There are three types of castings: First, create castings by knowledge from real-time web data. Second, castings are created by combining 13 kinds of scene and ontology. Last, recognizes the position of the player and 8 actions and combines them with the ontology to create castings. Data in [KBO(Korea Baseball Organization League)](https://www.koreabaseball.com "Korea Baseball League") games from 1 April 2018 to 14 April 2018 is used for learning three models.


## Demo
![Demo](/PNG/demo.gif)
<table>
<tr>
<td colspan="1"><sub>Demo 1</sub></td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/demo/demo1.gif?raw=1" height="500" width="800" alt="Noop"></td>
</tr>

<tr>
<td colspan="1"><sub>Demo 2</sub></td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/demo/demo2.gif?raw=1" height="500" width="800" alt="Noop"></td>
</tr>
</table>
<table>
<tr>
<td colspan="1"><sub>Ground to 1st Base</sub></td>
<td colspan="1"><sub>Double Play</sub></td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/demo/1st_ground.gif?raw=1" height="250" width="400" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/demo/double.gif?raw=1" height="250" width="400" alt="Noop"></td>
</tr>

<tr>
<td colspan="1"><sub>Outfield Single Hit</sub></td>
<td colspan="1"><sub>Outfield Two-Base Hit</sub></td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/demo/outfield_1.gif?raw=1" height="250" width="400" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/demo/outfield_2.gif?raw=1" height="250" width="400" alt="Noop"></td>
</tr>

</table>

## Models
![Flow_chart](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/flow_chart.png?raw=1)

## Scene Classifier
<table>
<tr><td colspan="3"><strong>About. BatterBox</strong></td></tr>
<tr>
<td colspan="1"><sub>BatterBox</sub></td>
<td colspan="1"><sub>Batter</sub></td>
<td colspan="1"><sub>Close up</sub></td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/batterbox.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/batter.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/closeup.jpg?raw=1" height="148" width="100" alt="Noop"></td>
</tr>

<tr><td colspan="7"><strong>About. Field</strong></td></tr>
<tr>
<td colspan="1"><sub>1st. Base</sub></td>
<td colspan="1"><sub>2nd. Base</sub></td>
<td colspan="1"><sub>3rd. Base</sub></td>
<td colspan="1"><sub>Right Outfield</sub></td>
<td colspan="1"><sub>Center Outfield</sub></td>
<td colspan="1"><sub>Left Outfield</sub></td>
<td colspan="1"><sub>ShortStop</sub></td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/1stbase.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/2ndbase.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/3rdbase.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/rightoutfield.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/centeroutfield.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/leftoutfield.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/shortstop.jpg?raw=1" height="148" width="100" alt="Noop"></td>
</tr>

<tr><td colspan="7"><strong>Etc.</strong></td></tr>
<tr>
<td colspan="1"><sub>Coach</sub></td>
<td colspan="1"><sub>Gallery</sub></td>
<td colspan="1"><sub>etc.</sub></td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/coach.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/gallery.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/etc.jpg?raw=1" height="148" width="100" alt="Noop"></td>
</tr>
</table>

#### Zero-Shot Learning
![Scene](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/zero_shot.png?raw=1)

- using 0 3rd Base data, 0 Left OutField data.
- In Baseball Game, There are few data of 3rd Base and Right OutField than others. But I can train these two data with others. Almost Baseball Field have symmetrical characters. So I can get these data with flipping other training data.
- 3rd Base <-> 1st Base
- Right OutField <-> Left OutField

---

## Player Detector
<table>

<tr>
<td colspan="1"><sub>Pitcher</sub></td>
<td colspan="1"><sub>Batter</sub></td>
<td colspan="1"><sub>Catcher</sub></td>
<td colspan="1"><sub>Fielder</sub></td>
<td colspan="1"><sub>Referee</sub></td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/player/pitcher.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/player/batter.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/player/catcher.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/player/fielder.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/player/referee.jpg?raw=1" height="148" width="100" alt="Noop"></td>
</tr>
</table>

---


## Motion Classifier

<table>

<tr>
<td colspan="1"><sub>Pitching</sub></td>
<td colspan="1"><sub>Waiting</sub></td>
<td colspan="1"><sub>Batting</sub></td>
<td colspan="1"><sub>Catching - Catcher</sub></td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/pitching.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/waiting.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/batting.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/catching-catcher.jpg?raw=1" height="148" width="100" alt="Noop"></td>
</tr>

<tr>
<td colspan="1"><sub>Throwing</sub></td>
<td colspan="1"><sub>Catching - Fielder</sub></td>
<td colspan="1"><sub>Walking</sub></td>
<td colspan="1"><sub>Running</sub></td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/throwing.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/catching-fielder.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/walking.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/running.jpg?raw=1" height="148" width="100" alt="Noop"></td>
</tr>

</table>

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
