import socket
import time
import json

with socket.socket() as sock:
    sock.connect(("169.254.119.30", 8080))
    a = 0
    while(True):
        if a % 3 == 0:
            content = {"text":  "홈팀이 유리한 가운데, 윌슨 오늘 경기 2개의 포볼로 타자 출루 시켰습니다", "motion": "strike_1"}
        elif a % 3 == 1:
            content = {"text":  "홈팀이 유리한 가운데, 윌슨 오늘 경기 2개의 포볼로 타자 출루 시켰습니다", "motion": "strike_0"}
        elif a % 3 == 2:
            content = {"text":  "홈팀이 유리한 가운데, 윌슨 오늘 경기 2개의 포볼로 타자 출루 시켰습니다", "motion": "strike_-1"}
        content = json.dumps(content, ensure_ascii=False)
        print(content)
        sock.sendall(content.encode('utf-8'))
        time.sleep(5)
        a = a + 1
        