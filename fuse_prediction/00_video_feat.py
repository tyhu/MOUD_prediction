### extract features from _au files
### Ting-Yao Hu 2016.03

import sys
sys.path.append('/home2/tingyaoh/ML/PyAI/util')
from util_ml import *
import os
import math

def cert_feat_mat(certfn,start,end):
    frame_rate = 30
    certfile = file(certfn,'r')
    certfile.readline()
    certfile.readline()
    certfile.readline()
    featlst = []
    for line in certfile:
        featlst.append([float(l) for l in line.lower().split()[38:]])
    startidx = int(math.floor(start*frame_rate))
    endidx = int(math.ceil(end*frame_rate))
    featmat = np.array(featlst)[startidx:endidx]
    featmean = np.nanmean(featmat,axis=0).tolist()
    featstd = np.nanstd(featmat,axis=0).tolist()

    #return featmean+featstd
    return featmean


labfn = '../MultimodalFeatures/audioFeatures/cats.txt'
cert_dir = '../MultimodalFeatures/visualFeaturesCERT/'
time_lab_fn = 'time_label.txt'
outfn = 'video_feat.txt'
labdict = {'neutral':0,'positive':1,'negative':2}

featdict = {}
lendict = {}
for line in file(time_lab_fn):
    lab = labdict[line.split()[1]]
    start = float(line.split()[2]) ### in sec
    end = float(line.split()[3])
    vidx = line.split()[0].split('_')[0]+'_'+line.split()[0].split('_')[1]
    uttidx = line.split()[0]
    certfn = cert_dir+vidx+'_au.txt'
    lendict[uttidx] = end-start
    if os.path.isfile(certfn):
        featmatlst = cert_feat_mat(cert_dir+vidx+'_au.txt',start,end)
        featsize = len(featmatlst)
        featdict[uttidx] = featmatlst

labfn = '../MultimodalFeatures/audioFeatures/cats.txt'
featlst = []
lablst = []
lenlst = []
labdict = {'neutral':0,'positive':1,'negative':2}
for line in file(labfn):
    feat = [0]*featsize
    uttidx = line.split()[0]
    if uttidx in featdict.keys():
        print uttidx
        feat = featdict[uttidx]
    featlst.append(feat)
    lenlst.append(lendict[uttidx])
X = np.array(featlst).astype('float')
l = np.array(lenlst)

X_bar = np.mean(X,axis=0)
for idx in range(X.shape[0]):
    if np.sum(X[idx])==0: X[idx] = X_bar
print X.shape
pickle.dump(X,open('video.pkl','wb'))
pickle.dump(l,open('length.pkl','wb'))

