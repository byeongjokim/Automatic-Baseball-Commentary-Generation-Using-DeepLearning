# -*- coding: utf-8 -*- 
import os
import csv
import cv2
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate import meteor_score
from _result.showandtell.im2txt.run_inference import test

def BLEU(generated_sentence, real_sentence):
    generated_sentence = [i.lower() for i in generated_sentence]
    real_sentence = [[j.lower() for j in i] for i in real_sentence]
    print("=====BLEU=====")
    generated_sentence_ = [sentence.split(" ") for sentence in generated_sentence]
    real_sentence_ = [[s.split(" ") for s in sentence] for sentence in real_sentence]

    total_bleu_1_score = 0.0
    total_bleu_2_score = 0.0
    total_bleu_3_score = 0.0
    total_bleu_4_score = 0.0
    total_bleu_5_score = 0.0

    total_bleu_1_score += round(corpus_bleu(real_sentence_, generated_sentence_, weights=tuple([1 / 1] * 1)), 4)
    total_bleu_2_score += round(corpus_bleu(real_sentence_, generated_sentence_, weights=tuple([1 / 2] * 2)), 4)
    total_bleu_3_score += round(corpus_bleu(real_sentence_, generated_sentence_, weights=tuple([1 / 3] * 3)), 4)
    total_bleu_4_score += round(corpus_bleu(real_sentence_, generated_sentence_, weights=tuple([1 / 4] * 4)), 4)

    print("==BLEU@1==")
    print(total_bleu_1_score)
    print("==BLEU@2==")
    print(total_bleu_2_score)
    print("==BLEU@3==")
    print(total_bleu_3_score)
    print("==BLEU@4==")
    print(total_bleu_4_score)

def meteor(generated_sentence, real_sentence):
    generated_sentence = [i.lower() for i in generated_sentence]
    real_sentence = [[j.lower() for j in i] for i in real_sentence]

    print("=====METEOR=====")
    total = 0.0
    for i in range(len(generated_sentence)):
        score = round(meteor_score.meteor_score(real_sentence[i], generated_sentence[i]), 4)
        total += score
    print(total/len(generated_sentence))

def for_examples():
    real_sentence = ["The pitcher gave up only 4 hits in 7 innings", "The hitter records 0.3 avg in this year", "This is 15 TBF of the pitcher"]
    generated_sentence = ["The pitcher Wilson, gave up 4 hits today", "The hitter Lee, records 0.3 avg in this season", "The pitcher has pitched in 15 TBF in this game"]
    
    generated_sentence_ = [sentence.split(" ") for sentence in generated_sentence]
    real_sentence_ = [[sentence.split(" ")] for sentence in real_sentence]
    
    for i in range(len(real_sentence_)):
        print(round(sentence_bleu(real_sentence_[i], generated_sentence_[i], weights=tuple([1 / 1] * 1)), 4))
        print(round(sentence_bleu(real_sentence_[i], generated_sentence_[i], weights=tuple([1 / 2] * 2)), 4))
        #print(round(sentence_bleu(real_sentence_[i], generated_sentence_[i], weights=tuple([1 / 3] * 3)), 4))
        #print(round(sentence_bleu(real_sentence_[i], generated_sentence_[i], weights=tuple([1 / 4] * 4)), 4))
        print(round(meteor_score.meteor_score([real_sentence[i]], generated_sentence[i]), 4))
        
def ours_eval():
    print("=========================OURS=========================")
    with open("_result/180906LGNC_FULL/resultwithreal.csv", "r") as f:
        generated_sentence = []
        real_sentence = []
        
        tmp_real_lines = []
        reader = csv.reader(f)
        count = 0
        for line in reader:
            tmp_real_lines.append(line)
        
        for line_num in range(len(tmp_real_lines)):
            if(tmp_real_lines[line_num][2] and tmp_real_lines[line_num][0] and tmp_real_lines[line_num + 1][1] and tmp_real_lines[line_num + 1][2]):
                generated_sentence.append(tmp_real_lines[line_num + 1][1])
                real_sentence.append([tmp_real_lines[line_num + 1][2]])
    
    BLEU(generated_sentence, real_sentence)
    meteor(generated_sentence, real_sentence)
    #meteor_mAP2(generated_sentence, real_sentence)

def show_and_tell_eval():
    print("=========================SHOW AND TELL=========================")
    with open("_result/180906LGNC_FULL/resultwithreal.csv", "r") as f:
        tmp_real_lines = []
        real_lines = []
        reader = csv.reader(f)
        count = 0
        for line in reader:
            tmp_real_lines.append(line)
        
        for line_num in range(len(tmp_real_lines)):
            if(tmp_real_lines[line_num][2] and tmp_real_lines[line_num][0]):
                real_lines.append({"frame":int(tmp_real_lines[line_num][0]), "sentence":tmp_real_lines[line_num+1][2], "image":"_result/180906LGNC_FULL/" + tmp_real_lines[line_num][0] + ".jpg"})
        
        # if (line[2] and line[1] and not line[0]):
        #     count = count + 1
        #     real_lines.append({"frame":int(line[0]), "sentence":line[2], "image":"_result/180906LGNC_FULL/" + line[0] + ".jpg"})

    image_files = [i["image"] for i in real_lines]
    image_files = ",".join(image_files)
    real_sentence = [[i["sentence"]] for i in real_lines]
    generated_sentence = test("_result/showandtell/im2txt/model/model.ckpt-5000000", "_result/showandtell/im2txt/model/word_counts.txt", image_files)
    print(generated_sentence)
    BLEU(generated_sentence, real_sentence)
    meteor(generated_sentence, real_sentence)

def s2vt_eval():
    print("=========================S2VT=========================")
    startframe_1 = 125634
    startframe_2 = 153296
    startframe_3 = 162557

    folder = "./_result/s2vt/"
    filenames = os.listdir(folder)

    output_file = []
    for filename in filenames:
        if "output" in filename:
            output_file.append({"videoID":int(filename.split("-")[0].split("_")[-1]), "index": int(filename.split("-")[1]), "filename":os.path.join(folder, filename)})
    output_file = sorted(output_file, key=lambda k:(k["videoID"], k["index"]))

    generated_lines = []
    for i in output_file:
        f = open(i["filename"], "r")
        while True:
            line = f.readline()
            if not line: break

            startframe = 5 * 29.97 * (i["index"] - 1) + int(line.split("\t")[0][3]) * 30
            if i["videoID"] == 1:
                startframe += startframe_1
            elif i["videoID"] == 2:
                startframe += startframe_2
            else:
                startframe += startframe_3
            endframe = startframe + 100

            generated_lines.append({"videoID":i["videoID"], "index":i["index"], "vidID":line.split("\t")[0], "generated":line.split("\t")[1].rstrip(), "startFrame":int(startframe), "endFrame":int(endframe)})

    generated_lines = sorted(generated_lines, key=lambda k:(k["videoID"], k["index"], k["vidID"]))

    with open("_result/180906LGNC_FULL/resultwithreal.csv", "r") as f:
        tmp_real_lines = []
        real_lines = []
        reader = csv.reader(f)
        count = 0
        for line in reader:
            tmp_real_lines.append(line)
        
        for line_num in range(len(tmp_real_lines)):
            if(tmp_real_lines[line_num][2] and tmp_real_lines[line_num][0]):
                real_lines.append({"frame":int(tmp_real_lines[line_num][0]), "sentence":tmp_real_lines[line_num+1][2]})
        
    real_sentence = []
    generated_sentence = []
    generated_frame = [i["endFrame"] for i in generated_lines]
    for real in real_lines:
        close_i = -123
        for i in generated_frame:
            if abs(real["frame"] - close_i) > abs(real["frame"] - i):
                close_i = i
        for j in generated_lines:
            if j["endFrame"] == close_i:
                generated_sentence.append(j["generated"])
                break
        real_sentence.append([real["sentence"]])

    BLEU(generated_sentence, real_sentence)
    meteor(generated_sentence, real_sentence)

def scst_eval():
    print("=========================SCST=========================")
    with open("_result/180906LGNC_FULL/resultwithreal.csv", "r") as f:
        tmp_real_lines = []
        real_lines = []
        reader = csv.reader(f)
        count = 0
        for line in reader:
            tmp_real_lines.append(line)
        
        for line_num in range(len(tmp_real_lines)):
            if(tmp_real_lines[line_num][2] and tmp_real_lines[line_num][0]):
                real_lines.append({"frame":int(tmp_real_lines[line_num][0]), "sentence":tmp_real_lines[line_num+1][2], "image":tmp_real_lines[line_num][0] + ".jpg"})
    
    with open("_result/scst/result.csv", "r") as f:
        all_generated_sentence = []
        rdr = csv.reader(f)
        for line in rdr:
            all_generated_sentence.append(line)

    generated_sentence = []
    for real in real_lines:
        for gen in all_generated_sentence:
            if(gen[0] == real["image"]):
                generated_sentence.append(gen[1])
                break

    real_sentence = [[i["sentence"]] for i in real_lines]
    BLEU(generated_sentence, real_sentence)
    meteor(generated_sentence, real_sentence)

def dvc_eval():
    print("=========================DVC=========================")
    with open("_result/180906LGNC_FULL/resultwithreal.csv", "r") as f:
        tmp_real_lines = []
        real_lines = []
        reader = csv.reader(f)
        count = 0
        for line in reader:
            tmp_real_lines.append(line)
        
        for line_num in range(len(tmp_real_lines)):
            if(tmp_real_lines[line_num][2] and tmp_real_lines[line_num][0]):
                real_lines.append({"frame":int(tmp_real_lines[line_num][0]), "sentence":tmp_real_lines[line_num+1][2]})
    
    with open("_result/DVC/result.csv", "r") as f:
        all_generated_sentence = []
        rdr = csv.reader(f)
        for line in rdr:
            all_generated_sentence.append(line)
    
    sentences = []
    for gen in all_generated_sentence:
        real_sentences = []
        for real in real_lines:
            if(int(gen[0].split(".")[0]) < real["frame"] and int(gen[1].split(".")[0]) > real["frame"]):
                real_sentences.append(real["sentence"])
        if(real_sentences):
            sentences.append({"generated":gen[2], "real":real_sentences})
    
    BLEU([i["generated"] for i in sentences], [i["real"] for i in sentences])
    meteor([i["generated"] for i in sentences], [i["real"] for i in sentences])

def wsdec_eval():
    print("=========================DVC=========================")
    with open("_result/180906LGNC_FULL/resultwithreal.csv", "r") as f:
        tmp_real_lines = []
        real_lines = []
        reader = csv.reader(f)
        count = 0
        for line in reader:
            tmp_real_lines.append(line)
        
        for line_num in range(len(tmp_real_lines)):
            if(tmp_real_lines[line_num][2] and tmp_real_lines[line_num][0]):
                real_lines.append({"frame":int(tmp_real_lines[line_num][0]), "sentence":tmp_real_lines[line_num+1][2]})
    
    with open("_result/WSDEC/data/custom/result.csv", "r") as f:
        all_generated_sentence = []
        rdr = csv.reader(f)
        for line in rdr:
            all_generated_sentence.append(line)
    
    sentences = []
    for gen in all_generated_sentence:
        real_sentences = []
        for real in real_lines:
            if(int(gen[0].split(".")[0]) < real["frame"] and int(gen[1].split(".")[0]) > real["frame"]):
                real_sentences.append(real["sentence"])
        if(real_sentences):
            sentences.append({"generated":gen[2], "real":real_sentences})
    
    BLEU([i["generated"] for i in sentences], [i["real"] for i in sentences])
    meteor([i["generated"] for i in sentences], [i["real"] for i in sentences])

# ours_eval()
# show_and_tell_eval()
# s2vt_eval()
# scst_eval()
# dvc_eval()
wsdec_eval()

# for_examples()