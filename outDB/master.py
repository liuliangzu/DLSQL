import numpy as np
import socket
import threading
def recv(id):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 32000+id
    server_address = ('172.17.0.3', port)
    client_socket.bind(server_address)
    client_socket.listen(1)
    socket, address = client_socket.accept()
    data = client_socket.recv(1000000)
    recv_list = eval(data.decode())

    client_socket.close()
    socket.close()
    return recv_list

def send():
    pass

result_list = []
threading_dict = dict()
for seg_id in range(1):
    thread = threading.Thread(target=recv, args=(seg_id),name='training_{}'.format(seg_id))
    threading_dict[seg_id] = thread
    thread.start()
    for thread_id, thread in threading_dict.items():
        thread.join()
        result_list.append(thread.result)
        del threading_dict[thread_id]

print(result_list)
