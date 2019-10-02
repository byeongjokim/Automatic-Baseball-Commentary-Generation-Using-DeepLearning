import socket

BUF_SIZE = 1024
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 8080))
    sock.listen()
    conn, addr = sock.accept()

    while True:
        data = conn.recv(BUF_SIZE)
        msg = data.decode()
        print(msg)
