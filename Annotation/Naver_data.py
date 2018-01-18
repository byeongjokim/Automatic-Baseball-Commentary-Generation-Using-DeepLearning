# -*- coding: utf-8 -*-
import json
import operator
import time

class NaverData():
    threshold = 6

    def __init__(self, filename, Resources):
        self.Resources = Resources

        data_file = open(filename, "rt", encoding="UTF8")
        data = json.load(data_file)
        data_file.close()
        self.game_info = data["gameInfo"]
        self.awayTeamLineUp = data["awayTeamLineUp"]
        self.homeTeamLineUp = data["homeTeamLineUp"]
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

        s = int(h1) * 3600 + int(m1)*60 + int(s1)

        h2 = end[:2]
        m2 = end[2:4]
        s2 = end[4:]

        e = int(h2) * 3600 + int(m2) * 60 + int(s2)

        return e-s

    def get_timeadd(self, start, end):
        h1 = start[:2]
        m1 = start[2:4]
        s1 = start[4:]

        s = int(h1) * 3600 + int(m1)*60 + int(s1)

        h2 = end[:2]
        m2 = end[2:4]
        s2 = end[4:]

        e = int(h2) * 3600 + int(m2) * 60 + int(s2)

        return e+s

    def secondTotime(self, sec):
        h = format(sec // 3600, '02')
        m = format((sec % 3600) // 60, '02')
        s = format(sec % 60, '02')
        sec = h + m + s
        return sec

    def calculate_start(self, count_delta, fps, o_start):
        start = int(count_delta/fps)
        start = self.secondTotime(start)
        start = self.get_timeadd(o_start, start)
        start = self.secondTotime(start)

        for i in self.relayTexts:
            if(int(i["pitchId"].split("_")[-1]) > int(start)):
                break

        no = i["seqno"] - 1
        return start, no

    def return_seq(self, start, no):

        relayText = self.relayTexts[no:]

        #start = "183122"

        count = 0
        pre_pitchId = '000000'
        tmp_pitchId = '000000'

        for i in relayText:
            self.now_relayText = i
            wait = 0
            if(count == 0):
                now = start

            else:
                pre_pitchId = now
                now = i["pitchId"].split("_")[-1]

                if(now != '-1'):
                    if(pre_pitchId == '-1'):
                        wait = self.get_timedelta(now, tmp_pitchId)

                    else:
                        wait = self.get_timedelta(now, pre_pitchId)


                    if (wait > self.threshold):
                        time.sleep(self.threshold)
                        self.Resources.set_long(True)
                        time.sleep(wait-self.threshold)
                    else:
                        time.sleep(wait)

                else: #now == -1
                    if(pre_pitchId != '-1'):
                        tmp_pitchId = pre_pitchId
                    wait = 0

            print(i["liveText"])

            count = count + 1
