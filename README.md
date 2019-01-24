# Baseball Game Casting with Deep Learning
There have been many studies to make captions or explanations about images and video with deep learning. A combination of CNN and RNN is mainly used to generate captions from the features of image and frame. Recently, the attention model has been combined to explain a more detailed and accurate situation. Through these studies, deep learning can describe the real state of an image Human-Like. However, when casting a sports event, castings are created not only from contextual information about who does what, but also from understanding and reasoning using the player’s information and past knowledge.<br>
This research describes the three models (scene classifier, player detection, motion recognition) to obtain contextual information from the sports frame and the ontology to inference knowledge from past data. There are three types of castings: First, create castings by knowledge from real-time web data. Second, castings are created by combining 13 kinds of scene and ontology. Last, recognizes the position of the player and 8 actions and combines them with the ontology to create castings. Data in [KBO(Korea Baseball Organization League)](https://www.koreabaseball.com "Korea Baseball League") games from 1 April 2018 to 14 April 2018 is used for learning three models.


## Demo
<table>
<tr>
<td colspan="1">Demo 1</td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/demo/demo1.gif?raw=1" height="500" width="800" alt="Noop"></td>
</tr>

<tr>
<td colspan="1">Demo 2</td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/demo/demo2.gif?raw=1" height="500" width="800" alt="Noop"></td>
</tr>
</table>
<table>
<tr>
<td colspan="1">Ground to 1st Base</td>
<td colspan="1">Double Play</td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/demo/1st_ground.gif?raw=1" height="250" width="400" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/demo/double.gif?raw=1" height="250" width="400" alt="Noop"></td>
</tr>

<tr>
<td colspan="1">Outfield Single Hit</td>
<td colspan="1">Outfield Two-Base Hit</td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/demo/outfield_1.gif?raw=1" height="250" width="400" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/demo/outfield_2.gif?raw=1" height="250" width="400" alt="Noop"></td>
</tr>

</table>

## Flow
![Flow_chart](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/flow.png?raw=1)

<table>
<tr>
<td colspan="1">Type</td>
<td colspan="1">Method</td>
<td colspan="1">Example</td>
<td colspan="1">in English</td>
</tr>
<tr>
<td colspan="1">A</td>
<td colspan="1">Casting Using <span style="color:blue"><strong>Web Data</strong></span> and <span style="color:red"><strong>Ontology(knowledge)</strong></span></td>
<td colspan="1"><span style="color:blue">스트라이크! 구석에 꽂히는 직구에</span> 타자가 제대로 속았습니다. <span style="color:red">유희관 투수 이번 시즌 높은 확률로 초구 스트라이크 잡아내고 있습니다.</span></td>
<td colspan="1"><span style="color:blue">Strike! The fastball in the corner.</span> <span style="color:red">Yoo Hee-kwan is making his first pitch this season with a high chance.</span></td>
</tr>
<tr>
<td colspan="1">B</td>
<td colspan="1">Casting Using <span style="color:blue"><strong>Scene Data</strong></span> and <span style="color:red"><strong>Ontology(knowledge)</strong></span></td>
<td colspan="1"> <span style="color:blue">오지환 타자와 유희관 투수 사이에</span> 긴장이 흐르는 가운데 오지환 타자 <span style="color:red">이번 시즌 2할 7푼의 타율을 기록하고 있습니다.</span></td>
<td colspan="1">Amid tensions <span style="color:blue">between Oh and Yoo</span>, <span style="color:red">Oh has recorded a .27 batting average this season.</span></td>
</tr>
<tr>
<td colspan="1">C</td>
<td colspan="1">Casting Using <span style="color:blue"><strong>Motion</strong></span> and <span style="color:green"><strong>Position</strong></span> of Player, and <span style="color:red"><strong>Ontology(knowledge)</strong></span></td>
<td colspan="1"><span style="color:green">유희관 투수</span> <span style="color:blue">공을 던졌습니다.</span></td>
<td colspan="1"><span style="color:green">Yoo Hee-kwan</span> <span style="color:blue">pitched.</span>
</td>
</tr>
</table>


## Models
### Scene Classifier
#### Model
![Scene Model](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/model.png?raw=1)
#### Classes
<table>
<tr><td colspan="3"><strong>About. BatterBox</strong></td></tr>
<tr>
<td colspan="1">BatterBox</td>
<td colspan="1">Batter</td>
<td colspan="1">Close up</td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/batterbox.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/batter.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/closeup.jpg?raw=1" height="148" width="100" alt="Noop"></td>
</tr>

<tr><td colspan="7"><strong>About. Field</strong></td></tr>
<tr>
<td colspan="1">1st. Base</td>
<td colspan="1">2nd. Base</td>
<td colspan="1">3rd. Base</td>
<td colspan="1">Right Outfield</td>
<td colspan="1">Center Outfield</td>
<td colspan="1">Left Outfield</td>
<td colspan="1">ShortStop</td>
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
<td colspan="1">Coach</td>
<td colspan="1">Gallery</td>
<td colspan="1">etc.</td>
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

### Player Detector
#### Model
[Yolo tiny v2 model](https://pjreddie.com/darknet/yolo/)
![Yolo Model](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/player/model.png?raw=1)
#### Classes
<table>
<tr>
<td colspan="1">Pitcher</td>
<td colspan="1">Batter</td>
<td colspan="1">Catcher</td>
<td colspan="1">Fielder</td>
<td colspan="1">Referee</td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/player/pitcher.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/player/batter.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/player/catcher.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/player/fielder.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/player/referee.jpg?raw=1" height="148" width="100" alt="Noop"></td>
</tr>
</table>

#### Recognize Position
<table>
<tr>
<td colspan="2">ex) Scene of 1st Base</td>
</tr>
<tr>
<td colspan="1">Detected <strong>one</strong> field player.</td>
<td colspan="1">Detected <strong>many</strong> field players.</td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/player/position_1.png?raw=1" height="200" width="380" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/player/position_2.png?raw=1" height="200" width="380" alt="Noop"></td>
</tr>

<tr>
<td colspan="1"><strong>Only one</strong> player who is in 1st Base scene<br>=> 1st Baseman</td>
<td colspan="1">The player who is in <strong>significant area</strong> of 1st Base scene<br>=> 1st Baseman</td>
</tr>
</table>

---


### Motion Classifier
#### Model
![Motion Model](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/model.png?raw=1)
#### Classes
<table>
<tr>
<td colspan="1">Pitching</td>
<td colspan="1">Waiting</td>
<td colspan="1">Swing</td>
<td colspan="1">Catching - Catcher</td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/pitching.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/waiting.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/batting.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/catching-catcher.jpg?raw=1" height="148" width="100" alt="Noop"></td>
</tr>

<tr>
<td colspan="1">Throwing</td>
<td colspan="1">Walking</td>
<td colspan="1">Running</td>
<td colspan="1">Catching - Fielder</td>
</tr>

<tr>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/throwing.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/walking.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/running.jpg?raw=1" height="148" width="100" alt="Noop"></td>
<td colspan="1"><img src="https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/catching-fielder.jpg?raw=1" height="148" width="100" alt="Noop"></td>
</tr>

</table>

---

### Ontology
#### Schema
![schema](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/ontology/schema.png?raw=1)

#### Example of retrieve knowledges
<table>
<tr>
<td colspan="1">Knowledge</td>
<td colspan="1">SPARQL</td>
</tr>
<tr>
<td colspan="1">Previous record between pitcher A and batter B</td>
<td colspan="1">
SELECT ?o WHERE {?s toHitter A . ?s fromPitcher B . ?s result ?o}
</td>
</tr>
</table>

---

### Web Data
![web](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/web/webdata.png?raw=1)

![web2ontology](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/web/web2ontology.png?raw=1)

- In [N company](http://www.naver.com), there is a [KBO Text Broadcasting site](http://sports.news.naver.com/kbaseball/schedule/indexnhn).
- Each events can intelligible with Ontology in real-time.

---

## How to run