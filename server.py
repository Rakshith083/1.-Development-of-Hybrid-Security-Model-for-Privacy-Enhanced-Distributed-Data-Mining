import threading
import socket

HEADERSIZE=100
host = '127.0.0.1'
port = 60000
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((host, port))
server.listen()
clients = []

def broadcast(message, client):
    for c in clients:
        if c == client:
            continue

        c.send(message)

def handle_client(client):
    while True:
        try:
            message=client.recv(1024)

            broadcast(message, client)

        except Exception as e:
            print(e)
            clients.remove(client)
            client.close()
            break


def receive():
    while True:
        print('Server is running and listening ...')
        client, address = server.accept()
        print(f'connection is established with {str(address)}')      
        
        clients.append(client)

        thread = threading.Thread(target=handle_client, args=(client,))
        thread.start()

if __name__ == "__main__":
    receive()
