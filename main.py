# -*- coding: utf-8 -*-
import json
import operator

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

    def return_seq(self):

        count = 0
        for i in self.relayTexts:

            if(count == 0):
                now = 183122
                pre_pitchId = 0
                tmp_pitchId = 0
            else:
                pre_pitchId = now
                now = i["pitchId"].split("_")[-1]

                if(now != '-1'):
                    if(pre_pitchId == '-1'):
                        print(int(now) - int(tmp_pitchId))

                    else:
                        print(int(now) - int(pre_pitchId))
                else: #now == -1
                    if(pre_pitchId != '-1'):
                        tmp_pitchId = pre_pitchId

            count = count + 1


t = TextData()
t.return_seq()