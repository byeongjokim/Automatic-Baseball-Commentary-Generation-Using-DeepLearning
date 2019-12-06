# Automatic-sports-commentary-framework
![ALL Model](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/all_models.png?raw=1)
Video captioning is known as a method to summarize or explain a video. However, it is very difficult to create sports commentaries, which are running scene-by-scene descriptions, for sports videos by using conventional video captioning. This is the case because sports commentary requires not only specific and varied information about every scene, such as player action descriptions in baseball, but also background knowledge and dynamic at-bats statistics that are not found in the video. We propose a new system to automatically generate commentary for baseball games. In our system, given real-time baseball videos, suitable descriptions are relayed using four deep-learning models (i.e., a scene classifier, player detector, motion recognizer, and pitching result recognizer) integrated with domain ontology. Using these four deep-learning models, pieces of information about “who is doing what in which area of the field” and “what results are expected” are obtained. This approach is used to select an appropriate template, which is combined with baseball ontology knowledge for the generation of commentary. We evaluate our system using baseball games from the [KBO (Korea Baseball Organization)](https://www.koreabaseball.com "Korea Baseball League") League’s 2018 season. As a result of the evaluation, we identify that our system can act as a commentator for baseball games.

## Demo
|[New] Demo<br/>(with NAO Robot, 5:11)|[New] Demo<br/>(with English, 6:47)|[Old] Demo 1<br/>(with pitching result recognition, 2:27)|
|----|----|----|
|[Onedrive](https://1drv.ms/v/s!AtL3vScJgk8Fkcwa8KMVNXRZuUuLsg?e=KLqqg2)|[Onedrive](https://1drv.ms/v/s!AtL3vScJgk8Fkcpz6SoHPrOEL9RoUg?e=MV7mN3)|[Onedrive](https://1drv.ms/v/s!AtL3vScJgk8Fkcp0HP8U7eliRSV5Gg?e=vsZpS5)|

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
Please wait a moment because the capacity of gif is large.

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
<td colspan="1">Casting using <span style="color:blue"><strong>*Web Data</strong></span>, <span style="color:green"><strong>Pitching Result Recognition</strong></span> and <span style="color:red"><strong>Ontology(knowledge)</strong></span></td>
<td colspan="1"><span style="color:blue"><span style="color:green">볼 인가요?</span> 스트라이크! 구석에 꽂히는 직구에</span> 타자가 제대로 속았습니다. <span style="color:red">유희관 투수 이번 시즌 높은 확률로 초구 스트라이크 잡아내고 있습니다.</span></td>
<td colspan="1"><span style="color:blue"><span style="color:green">Is it Ball?</span> Strike! The fastball in the corner.</span> <span style="color:red">Yoo Hee-kwan is making his first pitch this season with a high chance.</span></td>
</tr>
<tr>
<td colspan="1">B</td>
<td colspan="1">Casting using <span style="color:blue"><strong>Scene Data</strong></span> and <span style="color:red"><strong>Ontology(knowledge)</strong></span></td>
<td colspan="1"> <span style="color:blue">오지환 타자와 유희관 투수 사이에</span> 긴장이 흐르는 가운데 오지환 타자 <span style="color:red">이번 시즌 2할 7푼의 타율을 기록하고 있습니다.</span></td>
<td colspan="1">Amid tensions <span style="color:blue">between Oh and Yoo</span>, <span style="color:red">Oh has recorded a .27 batting average this season.</span></td>
</tr>
<tr>
<td colspan="1">C</td>
<td colspan="1">Casting using <span style="color:blue"><strong>Motion</strong></span> and <span style="color:green"><strong>Position</strong></span> of Player, and <span style="color:red"><strong>Ontology(knowledge)</strong></span></td>
<td colspan="1"><span style="color:green">유희관 투수</span> <span style="color:blue">공을 던졌습니다.</span></td>
<td colspan="1"><span style="color:green">Yoo Hee-kwan</span> <span style="color:blue">pitched.</span>
</td>
</tr>
</table>
** The referee's judgment(strike, ball, foul, out ...) is casted by web data

## Models
### Scene Classification
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

#### Data Augmentation
![Scene](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/scene/zero_shot.png?raw=1)

It is difficult to get a similar amount of scene data for each class because of the relatively infrequent number of plays near third base and right field. To solve this problem, we take advantage of the symmetric features of a baseball stadium. The flipped image of the 1st base area and the image of the 3rd base area look alike, despite the different stadiums. We supplement insufficient data for third base and right field by flipping first base and left field data. This is also applied in the opposite direction.

---

### Player Detection
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

### Motion Recognition
#### Model
![Motion Model](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/model.png?raw=1)
![Motion Model2](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/motion/model2.png?raw=1)

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

### Pitching Result Recognition
#### Model
![PRR Model](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/situation/model.png?raw=1)
![PRR Model2](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/situation/model2.png?raw=1)

#### Classes
- Strike
- Ball
- Foul
- Hit
- Ground out
- Fly out
- Etc.

---

### Ontology
#### Schema
![schema](https://github.com/byeongjokim/Baseball-Casting-with-Deep-Learning/blob/master/PNG/ontology/schema.png?raw=1)

#### Example of retrieve knowledges
<table>
  <tr>
    <th>Knowledge</th>
    <th>the recent record between the hitter A and the pitcher B</th>
  </tr>
  <tr>
    <td>Query</td>
    <td>SELECT ?date ?stadium ?inning ?result WHERE {?s ?toHitter ?A . ?s ?fromPitcher ?B . ?s ?hasTimeStamp ?t . ?s ?hasResult ?result . ?s ?inInning ?inning . ?s ?inGame ?g . ?g ?inDate ?date . ?g ?hasHomeTeam ?h . ?h ?hasHomeStadium ?stadium} order by desc(?t)</td>
  </tr>
  <tr>
    <td>Template</td>
    <td>KOR) B 투수와 A 타자의 최근 전적은 {date} {stadium}에 진행한 경기 {inning}회 {result}입니다
    <br><br>ENG) The batter "A" {result} in the {inning} inning of the latest game against the pitcher "B" on {date} at {stadium}
</td>
  </tr>
  <tr>
    <td>Commentary</td>
    <td>KOR) B 투수와 A 타자의 최근 전적은 2018년 10월 10일, 잠실에 진행한 경기 8회 스트라이크 아웃입니다
<br><br>ENG) The batter "A" struck out in the 8th inning of the latest game against the pitcher "B" on October 10th, 2018 at Jamsil
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

```
#download pretrained four models

mkdir _model
cd _model
wget -O https://1drv.ms/u/s!AtL3vScJgk8Fkcpcdo4SbT09PKJlvg?e=dq4uHV
unzip -j _model.zip
```
```
#download the web data files (about result "{videoname}.txt", about ball "{videoname}_ball.txt") to _data/{videoname}/
#download the video "{videoname}.mp4" to _data/{videoname}/

#ex) 2018. 09. 06 LG vs NC game
mkdir _data
cd _data
wget https://1drv.ms/u/s!AtL3vScJgk8FkcpryZAWuKuF18hiDQ?e=cOAJov
```
```
#modify GAME_DATE, GAME_NO, FILE_NAME in settings.py to find video and web data
#modify FIRST_BATTERBOX_START_IN_VIDEO, FIRST_BATTERBOX_START_IN_OWL, START_FRAME to make the same time point of the web data and the video
#you can set the FIRST_BATTERBOX_START_IN_VIDEO, FIRST_BATTERBOX_START_IN_OWL using frame number and pitchID of {videoname}.txt of the first pitching judgement 

vi settings.py
```
```
#run the Automatic-sports-commentary-framework
python kbo.py
```
```
#run the simulation with predicted bounding boxes and labels

vi kbo.py
#modify isSimulation=False to isSimulation=True

python kbo.py
```
```
#run the simulation with predicted bounding boxes and labels

vi kbo.py
#modify isSimulation=False and isAPI=True
#modify host and port in run_server method

python kbo.py
```