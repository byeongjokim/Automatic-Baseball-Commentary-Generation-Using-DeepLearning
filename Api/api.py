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

    def connect(self):
        self.sock = socket.socket()
        self.sock.connect((self.host, self.port))

        content = {"motion": "etc_0", "text": "중계방송 준비중입니다."}
        print(content)
        content = json.dumps(content, ensure_ascii=False)
        self.sock.sendall(content.encode('utf-8'))

    def relay(self, text, comment_type):
        motion = self.choose_motion(text, comment_type)

        content = {"text": text, "motion": motion}
        print(content)
        content = json.dumps(content, ensure_ascii=False)
        self.sock.sendall(content.encode('utf-8'))
