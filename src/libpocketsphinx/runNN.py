import os
import numpy as np
import socket
import struct
import time
import sys
def add_deltas(array):
    #check dimensions per frame
    dim = array.shape[1]
    num_frames = array.shape[0]
    #print "dim",dim
    #print "num_frames",num_frames
    delta=[]
    delta_rows=[]
    for i in range (num_frames):
        for j in range(dim):
            t1 = j-1
            t2 = j+1
            if t1 < 0:
                t1 = 0
            if t2 > dim-1:
                t2 = dim-1
        
            delta_tmp = array[i,t2]-array[i,t1]
            delta_rows.append(delta_tmp)
        delta.append(delta_rows)
        delta_rows= []
    delta = np.array(delta)
    return delta

#function to fuse delta and Acc into dataset
def fuse(dataset,delta_array,acc_array):
    new_dataset = []
    dim = dataset.shape[1]
    num_frames = dataset.shape[0]
    for i in range(num_frames):
        a = dataset[i]
        b = delta_array[i]
        c = acc_array[i]
        abc = np.concatenate((a,b,c),axis=0)
        new_dataset.append(abc)
    return np.array(new_dataset)

def predictFrame(model,frame,weight=1,offset=0):
	# frame = np.array([[[0 for i in range(75)] for j in range(20)]])
	# print "shape ",np.array(frame)
	x = frame[0]
	x=x[:10025]
	x=x.reshape((401,25))
	delta=add_deltas(x)
	acc=add_deltas(delta)
	x=np.array([fuse(x,delta,acc)])
	scores = model.predict(x)
	scores=scores[0]
	# n_active = scores.shape[1]
	# print freqs
	# scores /= freqs + (1.0 / len(freqs))
	scores = np.log(scores)/np.log(1.0001)
	scores *= -1
	scores -= np.min(scores,axis=1).reshape(-1,1)
	# scores = scores.astype(int)
	scores *= weight
	scores += offset
	truncateToShort = lambda x: 32676 if x > 32767 else (-32768 if x < -32768 else x)
	vf = np.vectorize(truncateToShort)
	scores = vf(scores)
	# print scores
	r_str = struct.pack('%sh' % len(scores[0]), *scores[0])

	# scores /= np.sum(scores,axis=0)
	return r_str

if __name__ == '__main__':
	print 'ARGS:',sys.argv
   	base, model_name, n_feats, acwt, port, cuda_id = sys.argv[:6]
	print  base, model_name, n_feats, acwt, port, cuda_id
	if cuda_id == -1:
		CUDA_VISIBLE_DEVICES = ''
	else:
		CUDA_VISIBLE_DEVICES = str(cuda_id)
	os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
	from keras.models import load_model

	#model_name = "/home/mshah1/GSOC/bestModels/best_CI.h5"
	HOST, PORT = '', int(port)
	listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	listen_socket.bind((HOST, PORT))
	listen_socket.listen(1)
	client_connection = None
	print 'Serving HTTP on port %s ...' % PORT
	while True:
		if client_connection == None:
		    client_connection, client_address = listen_socket.accept()
		    model = load_model(model_name)
		r = client_connection.recv(4)
		if len(r) != 4:
			print "Expected 4 bytes for PACKET_LEN got " + str(len(r))
			continue	
		packet_len = struct.unpack('i',r)[0]
		print packet_len
		full_req = ""
		while len(full_req) < packet_len:
			partial_req = client_connection.recv(packet_len)
			full_req += partial_req
		assert(len(full_req) == packet_len)
		
		frame = full_req
		frame = list(struct.unpack('%sf' % int(n_feats), frame))
		frame = np.array([frame])
		
		resp = str(predictFrame(model,frame,weight=float(acwt)))
		print time.time()
		client_connection.send(resp)
		print time.time()
