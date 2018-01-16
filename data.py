# -*- coding: utf-8 -*-
import json
import operator
import time
import csv

class TextData():
    def __init__(self, filename='./_data/20171030KIADUSAN.txt'):
        data_file = open(filename, "rt", encoding="UTF8")
        data = json.load(data_file)
        data_file.close()
        self.game_info = data["gameInfo"]
        self.relayTexts = self.sort_with_seqno(data["relayTexts"])

    def sort_with_seqno(self, relayTexts):
        newlist = []
        newlist = newlist + relayTexts['1']
        newlist = newlist + relayTexts['2']
        newlist = newlist + relayTexts['3']
        newlist = newlist + relayTexts['4']
        newlist = newlist + relayTexts['5']
        newlist = newlist + relayTexts['6']
        newlist = newlist + relayTexts['7']
        newlist = newlist + relayTexts['8']
        newlist = newlist + relayTexts['9']
        newlist = newlist + relayTexts['currentBatterTexts']
        newlist = newlist + [relayTexts['currentOffensiveTeam']]
        newlist = newlist + [relayTexts['currentBatter']]

        newlist.sort(key=operator.itemgetter("seqno"))
        #for i in newlist:
        #    print(i)

        return newlist

    def get_timedelta(self, end, start):
        h1 = start[:2]
        m1 = start[2:4]
        s1 = start[4:]

        s = int(h1)*3600 + int(m1)*60 + int(s1)

        h2 = end[:2]
        m2 = end[2:4]
        s2 = end[4:]

        e = int(h2) * 3600 + int(m2) * 60 + int(s2)

        return e-s

    def return_seq(self):

        count = 0
        pre_pitchId = '000000'
        tmp_pitchId = '000000'

        for i in self.relayTexts:
            wait = 0
            if(count == 0):
                now = '183122'
            else:
                pre_pitchId = now
                now = i["pitchId"].split("_")[-1]

                if(now != '-1'):
                    if(pre_pitchId == '-1'):
                        wait = self.get_timedelta(now, tmp_pitchId)

                    else:
                        wait = self.get_timedelta(now, pre_pitchId)
                else: #now == -1
                    if(pre_pitchId != '-1'):
                        tmp_pitchId = pre_pitchId
                    wait = 0

            if(wait > 60):
                time.sleep(wait / 2)
                self.vision_data()
                time.sleep(wait / 2)
            else:
                time.sleep(wait)

            print(i["liveText"])

            count = count + 1

    def vision_data(self):
        print("a")


class SceneData():
    def __init__(self, path, fps=29.970):
        print("sceneData")
        self.path = path
        self.fps = fps
        self.get_data_csv()

    def get_data_csv(self):
        f = open(self.path, 'r', encoding='utf-8')
        reader = csv.reader(f)
        result = []
        count = 1
        for line in reader:
            if not line:
                pass

            elif(line[0] == str(count)):
                result.append({"SceneNumber":count, "start":int(line[1]), "end":int(float(line[4])*self.fps) + int(line[1]) })
                count = count + 1

        self.result = result
        f.close()

    def prepare_data(self):





