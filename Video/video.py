import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from time import time
import settings
from Api.api import API

def play(resource):
    def _text_2_img(text):
        img = Image.new('RGB', (1080, 50), color=(180, 180, 180))
        font = ImageFont.truetype("gulim.ttc", 20)
        d = ImageDraw.Draw(img)
        d.text((10, 10), text, font=font, fill=(0, 0, 0))

        return np.asarray(img)

    video = cv2.VideoCapture(settings.VIDEO_FILE)
    video.set(1, settings.START_FRAME)
    fps = 1 / 29.97

    success, frame = video.read()
    h, w, c = frame.shape
    frameno = settings.START_FRAME + 1
    resource.set_frameno(frameno)

    text = resource.get_annotation()
    textimage = _text_2_img(text)

    while (cv2.waitKey(1) != ord('q')):
        start = time()
        frameno = frameno + 1

        success, frame = video.read()

        resource.set_frame(frame)
        resource.set_frameno(frameno)

        if (resource.is_new_annotation_video()):
            text = resource.get_annotation()
            textimage = _text_2_img(text)
            print(text)

        frame[h - 100 : h - 50, 100 : w - 100] = textimage
        cv2.imshow("Automatic Sports Commentary", frame)

        diff = time() - start
        while diff < fps:
            diff = time() - start

def play_bbox(frameno):
    def _cal_mean_iou(bbox1, bboxes2):
        s = 0.0
        for bbox2 in bboxes2:
            min_x = max(bbox1[0], bbox2[0])
            max_x = min(bbox1[2], bbox2[2])
            min_y = max(bbox1[1], bbox2[1])
            max_y = min(bbox1[3], bbox2[3])

            if (max_x < min_x or max_y < min_y):
                s = s + 0.0
            else:
                inter = (max_x - min_x) * (max_y - min_y)
                bb1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                bb2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

                iou = inter / (bb1 + bb2 - inter)
                if (iou >= 0.0 and iou <= 1.0):
                    s = s + iou
                else:
                    s = s + 0.0
        return s / len(bboxes2)

    def _get_player_seq(image, position_images_seq, position_images_bbox_seq, human_coordinates):
        image_h, image_w, image_c = image.shape

        pitcher = [human_coordinate[:4] for human_coordinate in human_coordinates if (human_coordinate[4] == "pitcher")]
        batter = [human_coordinate[:4] for human_coordinate in human_coordinates if (human_coordinate[4] == "batter")]
        player = [human_coordinate[:4] for human_coordinate in human_coordinates if (human_coordinate[4] == "player")]

        for p in pitcher:
            if(position_images_bbox_seq["pitcher"]):
                if(_cal_mean_iou(p, [position_images_bbox_seq["pitcher"][-1]]) > 0.3):
                    position_images_bbox_seq["pitcher"].append(p)
                    position_images_seq["pitcher"].append(image[p[1]:p[3], p[0]:p[2]])
            else:
                position_images_bbox_seq["pitcher"].append(p)
                position_images_seq["pitcher"].append(image[p[1]:p[3], p[0]:p[2]])

        for p in batter:
            if(position_images_bbox_seq["batter"]):
                if(_cal_mean_iou(p, [position_images_bbox_seq["batter"][-1]]) > 0.3):
                    position_images_bbox_seq["batter"].append(p)
                    position_images_seq["batter"].append(image[p[1]:p[3], p[0]:p[2]])
            else:
                position_images_bbox_seq["batter"].append(p)
                position_images_seq["batter"].append(image[p[1]:p[3], p[0]:p[2]])

        if (player):
            player.sort(key=lambda x: (pow(((x[0] + x[2])/2 - image_h/2), 2) + pow(((x[1] + x[3])/2 - image_w/2), 2)))
            p = player[0]
            if(position_images_bbox_seq["player"]):
                if (_cal_mean_iou(p, [position_images_bbox_seq["player"][-1]]) > 0.3):
                    position_images_bbox_seq["player"].append(p)
                    position_images_seq["player"].append(image[p[1]:p[3], p[0]:p[2]])
            else:
                position_images_bbox_seq["player"].append(p)
                position_images_seq["player"].append(image[p[1]:p[3], p[0]:p[2]])

        return position_images_seq, position_images_bbox_seq
    
    def _draw_frame_no(frame, frameno):
        image_h, image_w, image_c = frame.shape
        img = Image.new('RGB', (300, 110), color=(255, 255, 255))
        font = ImageFont.truetype("gulim.ttc", 20)
        d = ImageDraw.Draw(img)
        game_info = "2018 09 29 LG VS NC in Seoul\n\n"
        d.text((10, 20), game_info+frameno, font=font, fill=(0, 0, 0))

        frame[0 : 110, 0 : 300] = np.asarray(img)

    def _draw_scene_label(frame, scene_label):
        image_h, image_w, image_c = frame.shape
        img = Image.new('RGB', (400, 50), color=(0, 250, 0))
        font = ImageFont.truetype("gulim.ttc", 20)
        d = ImageDraw.Draw(img)
        d.text((10, 10), scene_label, font=font, fill=(0, 0, 0))

        frame[0 : 50, image_w-400 : image_w] = np.asarray(img)

    def _draw_situation_label(frame, situation_label):
        image_h, image_w, image_c = frame.shape
        img = Image.new('RGB', (400, 50), color=(250, 0, 0))
        font = ImageFont.truetype("gulim.ttc", 20)
        d = ImageDraw.Draw(img)
        d.text((10, 10), situation_label, font=font, fill=(0, 0, 0))

        frame[50 : 100, image_w-400 : image_w] = np.asarray(img)

    def _draw_key_player_motion(frame, bbox, motion_label, motion_label_to_word, position):
        if(motion_label is not None):
            motion_label = motion_label_to_word[motion_label]
        else:
            motion_label = "Etc"

        x, y, w, h = bbox
        image_h, image_w, image_c = frame.shape
        img = Image.new('RGB', (200, 50), color=(255, 255, 255))
        font = ImageFont.truetype("gulim.ttc", 15)
        d = ImageDraw.Draw(img)
        d.text((10, 10), position + ":\n" + motion_label, font=font, fill=(0, 0, 0))

        try:
            frame[y : y+50, w : w+200] = np.asarray(img)
        except ValueError:
            height, width, c = frame[y : y+50, w : w+200].shape
            frame[y : y+50, w : w+200] = np.asarray(img)[:height, :width]
       

    import tensorflow as tf
    from Vision.Model.scene import Scene_Model
    from Vision.Model.detect import Detect_Model
    from Vision.Model.motion import Motion_Model, CAE
    from Vision.Model.situation import Situation_Model

    sess = tf.Session()
    scene = Scene_Model(sess=sess)
    detect = Detect_Model(sess=sess)
    motion = Motion_Model(sess=sess)
    situation = Situation_Model(sess=sess)

    video = cv2.VideoCapture(settings.VIDEO_FILE)
    video.set(1, frameno)
    fps = 1 / 29.97

    success, image = video.read()
    image_h, image_w, image_c = image.shape

    scene_label_to_word = ["Batter's Box", "Batter", "Closeup", "Coach", "Gallery", "Frst Base", "Center Outfield", "Right Outfield", "Second Base", "Etc", "Third Base", "Left Outfield", "Short Stop"]
    motion_label_to_word = ["Batting", "Batting Waiting", "Throwing", "Pitching", "Catch: catcher", "Catch: fielder", "Run", "Walking", "Etc"]
    situation_label_to_word = ["Strike", "Ball", "Foul", "Hit", "Ground Out", "Flying Out", "Etc"]

    position_images_seq = {"pitcher": [], "batter": [], "player": []}
    position_images_bbox_seq = {"pitcher": [], "batter": [], "player": []}
    motion_label = {"pitcher": None, "batter": None, "player": None}
    
    situation_count = 0
    scene_queue = []
    scene_length = 0
    human_queue = []
    
    pre_scene_label = -1    

    while (cv2.waitKey(1) != ord('q')):        
        video.set(1, frameno)
        success, image = video.read()

        scene_label, scene_score, featuremap = scene.predict(image=image)

        if(scene_label != pre_scene_label):
            position_images_seq = {"pitcher": [], "batter": [], "player": []}
            position_images_bbox_seq = {"pitcher": [], "batter": [], "player": []}
            motion_label = {"pitcher":None, "batter":None, "player":None}
        
        human_coordinates = detect.predict(image=image)
        if(human_coordinates):
            position_images_seq, position_images_bbox_seq = _get_player_seq(image=image, position_images_seq=position_images_seq, position_images_bbox_seq=position_images_bbox_seq, human_coordinates=human_coordinates)
        
        if (scene_label == 0 and position_images_seq["pitcher"] and motion_label["pitcher"] != 3):
            pitcher_motion_label, motion_score = motion.predict(position_images_seq["pitcher"])
            motion_label["pitcher"] = pitcher_motion_label
            if(pitcher_motion_label == 3):
                situation_count = 1

        if (scene_label == 0 and position_images_seq["batter"] and motion_label["batter"] != 0):
            batter_motion_label, motion_score = motion.predict(position_images_seq["batter"])
            motion_label["batter"] = batter_motion_label

        if (scene_label != 0 and position_images_seq["player"] and motion_label["player"] == None):
            player_motion_label, motion_score = motion.predict(position_images_seq["player"])
            motion_label["player"] = player_motion_label

        _draw_situation_label(image, "Result: Before Pitching")
        if(situation_count > 0):
            _draw_situation_label(image, "Result: Before reaching Threshold")
            situation_count = situation_count + 1
            scene_queue.append(featuremap)
            scene_length = scene_length + 1
            if(human_coordinates):
                human = []
                for x, y, w, h, _ in human_coordinates[:6]:
                    human.append(cv2.resize(image[y:h, x:w], (50, 50)))
                human_queue.append(human)
            else:
                human_queue.append([])
            scene_queue, human_queue, situation_label, situation_score = situation.predict(scene_queue=scene_queue, human_queue=human_queue, L=scene_length)

            if (situation_count > 30):
                _draw_situation_label(image, "Result: " + situation_label_to_word[situation_label])

                situation_count = 0
                scene_queue = []
                scene_length = 0
                human_queue = []

        _draw_scene_label(image, "Scene: " + scene_label_to_word[scene_label])
        _draw_frame_no(image, "Frame No: " + str(frameno))
        if(human_coordinates):
            for x, y, w, h, _ in human_coordinates[:6]:
                image = cv2.rectangle(image, (x, y), (w, h), color=(0, 255, 255), thickness=2)

            if(position_images_seq["pitcher"]):
                x, y, w, h = position_images_bbox_seq["pitcher"][-1]
                image = cv2.rectangle(image, (x, y), (w, h), color=(0, 0, 255), thickness=2)
                _draw_key_player_motion(image, position_images_bbox_seq["pitcher"][-1], motion_label["pitcher"], motion_label_to_word, "pitcher")

            if(position_images_seq["batter"]):
                x, y, w, h = position_images_bbox_seq["batter"][-1]
                image = cv2.rectangle(image, (x, y), (w, h), color=(0, 0, 255), thickness=2)
                _draw_key_player_motion(image, position_images_bbox_seq["batter"][-1], motion_label["batter"], motion_label_to_word, "batter")

            if(position_images_seq["player"]):
                x, y, w, h = position_images_bbox_seq["player"][-1]
                image = cv2.rectangle(image, (x, y), (w, h), color=(0, 0, 255), thickness=2)
                _draw_key_player_motion(image, position_images_bbox_seq["player"][-1], motion_label["player"], motion_label_to_word, "player(in " + scene_label_to_word[scene_label] + ")")
        
        #cv2.imwrite("./"+str(frameno)+".jpg", image)
        cv2.imshow("Automatic Sports Commentary", image)
        
        frameno = frameno + 5
        pre_scene_label = scene_label

def play_API(resource, host, port):
    print("[+] activate getting frame from video and send commentary to NAO robot")
    api = API(resource=resource, host=host, port=port)
    api.connect()

    video = cv2.VideoCapture(settings.VIDEO_FILE)
    video.set(1, settings.START_FRAME)
    fps = 1 / 29.97

    success, frame = video.read()
    h, w, c = frame.shape
    frameno = settings.START_FRAME + 1
    resource.set_frameno(frameno)
    text = resource.get_annotation()

    while (cv2.waitKey(1) != ord('q')):
        start = time()
        frameno = frameno + 1

        success, frame = video.read()

        resource.set_frame(frame)
        resource.set_frameno(frameno)
        cv2.imshow("Automatic Sports Commentary", frame)

        if (resource.is_new_annotation_video()):
            text = resource.get_annotation()
            comment_type = resource.get_action()
            api.relay(text, comment_type)

        diff = time() - start
        while diff < fps:
            diff = time() - start