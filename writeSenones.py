import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-ctl', dest="ctl",help="full path to ctl file",required = True)
parser.add_argument('-eo', dest="eo",help="output file extension",required = True)
parser.add_argument('-ei', dest="ei",help="input file extension",required = True)
parser.add_argument('-feat', dest="feat",help="full path to feat directory",required = True)
parser.add_argument('-do', dest="do",help="path to output directory",required = True)
parser.add_argument('-v', dest="v",help="verbosity",action="store_true",default = False)
arguments = parser.parse_args()
ctl = arguments.ctl
feats =arguments.feat
ei=arguments.ei
eo=arguments.eo
do=arguments.do
v=arguments.v

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model
import numpy as np
import struct
import pylab as pl
from sklearn import preprocessing
#to turn off tf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import readlog

def readMFC(fname,nFeats):
	#will return feat vector with data+delta and Acc 
	data = []
	with open(fname,'rb') as f:
		v = f.read(4)
		head = struct.unpack('I',v)[0]
		v = f.read(nFeats * 4)
		while v:
			frame = list(struct.unpack('%sf' % nFeats, v))
			data .append(frame)
			v = f.read(nFeats * 4)
	data = np.array(data)
	assert(data.shape[0] * data.shape[1] == head)
	delta=add_deltas(data)
	acc=add_deltas(delta)
	data=fuse(data,delta,acc)
	return data

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

def writeSenScores(filename,scores,freqs,weight,offset):
	n_active = scores.shape[1]
	s = ''
	s = """s3
version 0.1
mdef_file ../../en_us.cd_cont_4000/mdef
n_sen 138
logbase 1.000100
endhdr
"""
	s += struct.pack('I',0x11223344)
	scores /= freqs + (1.0 / len(freqs))
	scores = np.log(scores)/np.log(1.0001)
	scores *= -1
	scores -= np.min(scores,axis=1).reshape(-1,1)
	# scores = scores.astype(int)
	scores *= weight
	scores += offset
	truncateToShort = lambda x: 32676 if x > 32767 else (-32768 if x < -32768 else x)
	vf = np.vectorize(truncateToShort)
	scores = vf(scores)
	# scores /= np.sum(scores,axis=0)
	for r in scores:
		# print np.argmin(r)
		s += struct.pack('h',n_active)
		r_str = struct.pack('%sh' % len(r), *r)
		s += r_str
	with open(filename,'w') as f:
		f.write(s)	

def getPredsFromArray(model,data,nFrames,filenames,res_dir,res_ext,freqs):
	preds = model.predict(data,verbose=1,batch_size=2048)
	pos = 0
	for i in range(len(nFrames)):
		fname = filenames[i][:-4]
		fname = reduce(lambda x,y: x+'/'+y,fname.split('/')[4:])
		stdout.write("\r%d/%d 	" % (i,len(filenames)))
		stdout.flush()
		res_file_path = res_dir+fname+res_ext
		dirname = os.path.dirname(res_file_path)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		writeSenScores(res_file_path,preds[pos:pos+nFrames[i]],freqs)
		pos += nFrames[i]

def getPredsFromFilelist(model,filelist,file_dir,file_ext,
							res_dir,res_ext,freqs,context_len=4,
							weight=1,offset=0):
	with open(filelist) as f:
		files = f.readlines()
		files = map(lambda x: x.strip(),files)
	filepaths = map(lambda x: file_dir+"/"+x+file_ext,files)
	scaler = preprocessing.StandardScaler(copy=False,with_std=False)
	for i in range(len(filepaths)):
		print "\n",i+1,"/",len(filepaths)
		if v:
			print "feature file:",filepaths[i]

		f = filepaths[i]
		if not os.path.exists(f):
			print "\n",f
			continue
		data = readMFC(f,25)
		data=np.array([data])
		if v:
			print "Prediction size:",data.shape
		#Since passing only single utterance, shouldn't need padding
		preds = model.predict(data,batch_size=data.shape[0])
		data_postproc_fn = lambda x: x[:,range(138)] / np.sum(x[:,range(138)], axis=1).reshape(-1,1)
		preds=data_postproc_fn(preds[0])
		res_file_path = res_dir+"/"+files[i]+res_ext
		if v:
			print "senfile:",res_file_path
		dirname = os.path.dirname(res_file_path)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		writeSenScores(res_file_path,preds,freqs,weight,offset)


model=load_model("model_4_138")
freq = np.load("state_freq_dev.npy")
print "Starting Evaluation.."
getPredsFromFilelist(model,ctl,feats,ei,do,eo,freq,context_len=1,weight=1)
print "Evaluation complete.Senfiles stored in output dir"
