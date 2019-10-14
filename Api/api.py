import socket

class API():
    def __init__(self, resource, host, port, cheering="home"):
        self.resource = resource
        self.host = host
        self.port = port

        self.cheering = cheering

    def choose_motion(self, text, comment_type):
        motions = ["스트라이크", "파울", "페어", "아웃", "세이프", "홈런"]

        good_attack = ["진루", "출루", "안타", "세이프", "홈런"]
        bad_attack = ["스트라이크", "파울", "아웃"]

        # 00 motions+negative
        # 01 motions+positive
        motion = None

        # if self.cheering == "home":
        #
        # elif self.cheering == "away":
        #
        # else:
        #

        return motion

    def relay(self):
        with socket.socket() as sock:
            sock.connect((self.host, self.port))

            while (True):
                if (self.resource.is_new_annotation_video()):
                    text = self.resource.get_annotation()
                    comment_type = self.resource.get_action()

                    motion = self.choose_motion(text, comment_type)

                    content = {"comment_type": comment_type, "text": text, "motion": motion}
                    print(str(content))
                    sock.sendall(str(content).encode())
