import socket

with socket.socket() as sock:
    sock.connect(("127.0.0.1", 9559))
    content = "asdAd"
    sock.sendall(str(content).encode())