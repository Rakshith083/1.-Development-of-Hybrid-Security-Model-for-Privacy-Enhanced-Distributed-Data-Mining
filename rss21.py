from phe import paillier
import pickle

HEADERSIZE = 100

global erecive
erecive={}
global epkfinal
epkfinal={}
global s
s = []
n=3
public_key , private_key = paillier.generate_paillier_keypair()


def si_send(si, client):
 
    # y = input("Please press enter in si")
    
    message = pickle.dumps(si)
    message = bytes(f"{len(message):<{HEADERSIZE}}", 'utf-8') + message
    client.send(message)

def si_receive(client):

    try:

        full_msg = b''
        new_msg = True
        while True:

            msg=client.recv(1024)

            if new_msg:
            
                msglen = int((msg[:HEADERSIZE]))
                new_msg = False

            full_msg += msg

            if len(full_msg)-HEADERSIZE == msglen:
                
                s.append(pickle.loads(full_msg[HEADERSIZE:]))

                break 
            
    except Exception as e:
        print(e)

def epk_receive(pid, client):

    try:
        full_msg = b''
        new_msg = True

        while True:

            msg=client.recv(1024)

            if new_msg:
            
                msglen = int((msg[:HEADERSIZE]))
                new_msg = False

            full_msg += msg
    

            if len(full_msg)-HEADERSIZE == msglen:
                
                pr=pickle.loads(full_msg[HEADERSIZE:])
                print('pr in epk',pr)
                for i in pr.keys():
                    if i[0]==pid:
                        print(i)
                        epkfinal[i]=pr[i]
                        print(epkfinal[i])
                        break
                break        

    except Exception as e:
        print(e)

def epk_send(rt, client):

    # y=input("Please press enter in epk")

    message = pickle.dumps(rt)
    message = bytes(f"{len(message):<{HEADERSIZE}}", 'utf-8')+ message
    client.send(message)

def client_receive(pid,client):

    try:
        full_msg = b''
        new_msg = True

        while True:

            msg=client.recv(1024)

            if new_msg:
            
                msglen = int((msg[:HEADERSIZE]))
                new_msg = False

            full_msg += msg

            if len(full_msg)-HEADERSIZE == msglen:
                
                pr=pickle.loads(full_msg[HEADERSIZE:])
                print()
    
                for i in pr.keys():
                    if i[1]==pid:
                        print(i)
                        erecive[i]=pr[i]
                        print(erecive[i])
                        break
                break

    except Exception as e:
        print(e)
    
    print('out of client')

def client_send(rt,client):

    # y=input("Please press enter in client")

    message = pickle.dumps(rt)
    message = bytes(f"{len(message):<{HEADERSIZE}}", 'utf-8')+ message
    client.send(message)