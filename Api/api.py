import socket
import json

class API():
    def __init__(self, resource, host, port, cheering="home"):
        self.resource = resource
        self.host = host
        self.port = port

        self.cheering = cheering

    def choose_motion(self, text, comment_type):
        # btop == 0 -> home attacking
        btop = self.resource.get_btop()

        motions = ["strike", "ball", "foul", "hit", "hits", "out", "homerun", "etc"]

        good_attack = ["hits", "hit", "ball", "homerun"]
        bad_attack = ["strike", "foul", "out"]
        nop = ["etc"]

        if self.cheering == "home":
            if btop == 0:
                # good -> positive
                # bad -> negative
                if comment_type in good_attack:
                    return comment_type+"_1"
                elif comment_type in bad_attack:
                    return comment_type+"_-1"
                else:
                    return comment_type+"_0"
            else:
                # good -> negative
                # bad -> positive
                if comment_type in good_attack:
                    return comment_type+"_-1"
                elif comment_type in bad_attack:
                    return comment_type+"_1"
                else:
                    return comment_type+"_0"
        else:
            if btop == 0:
                # good -> negative
                # bad -> positive
                if comment_type in good_attack:
                    return comment_type+"_-1"
                elif comment_type in bad_attack:
                    return comment_type+"_1"
                else:
                    return comment_type+"_0"
            else:
                # good -> positive
                # bad -> negative
                if comment_type in good_attack:
                    return comment_type+"_1"
                elif comment_type in bad_attack:
                    return comment_type+"_-1"
                else:
                    return comment_type+"_0"

    def relay(self):
        with socket.socket() as sock:
            sock.connect((self.host, self.port))

            content = {"motion": "etc_0", "text": "중계방송 준비중입니다."}
            print(content)
            content = json.dumps(content, ensure_ascii=False)
            sock.sendall(content.encode('utf-8'))

            while (True):
                if (self.resource.is_new_annotation_video()):
                    text = self.resource.get_annotation()
                    comment_type = self.resource.get_action()

                    motion = self.choose_motion(text, comment_type)

                    content = {"text": text, "motion": motion}
                    print(content)
                    content = json.dumps(content, ensure_ascii=False)
                    sock.sendall(content.encode('utf-8'))
