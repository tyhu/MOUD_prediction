### text feature extraction
### Ting-Yao Hu,2016.04

import sys
import os
import glob
import string
import math
sys.path.append('/home2/tingyaoh/ML/PyAI/util')
from util_ml import *

def parse_sphinx4_align(s):
    lst = s.split('{')
    tokenlst = []
    endtimelst = []
    for l in lst[1:]:
        token = l.split(',')[0]
        if token[0]=='<': continue
        endtime = max(float(l.split(',')[2].split(':')[0].split('[')[-1]),0)/1000
        tokenlst.append(token)
        endtimelst.append(endtime)
    return tokenlst,endtimelst

def prediction_feature(infn):
    tokenlst = []
    endtimelst = []
    for line in file(infn):
        lst = line.split('{')
        if len(lst)==1: continue
        tlst,elst = parse_sphinx4_align(line)
        tokenlst+=tlst
        endtimelst+=elst
    return tokenlst,endtimelst

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

    return featmean


labfn = '../MultimodalFeatures/audioFeatures/cats.txt'
featlst = []
lablst = []
labdict = {'neutral':0,'positive':1,'negative':2}
pred_sec_lst = range(1,11)

### audio
"""
for line in file(labfn):
    fn = '../audio_prediction/opensmile_feat/'+line.split()[0]+'_seq.feat'
    x = np.genfromtxt(fn,delimiter=';')
    featlst.append(x)

X = np.array(featlst).astype('float')
print X.shape
pickle.dump(X,open('audio_seq.pkl','wb'))

### video
cert_dir = '../MultimodalFeatures/visualFeaturesCERT/'
time_dict = {}
for line in file('../video_prediction/time_label.txt'):
    lst = line.split()
    time_dict[lst[0]]=(float(lst[2]),float(lst[3]))

totalvlst = []
for line in file(labfn):
    uttidx = line.split()[0]
    lst = line.split()[0].split('_')
    vidx = lst[0]+'_'+lst[1]
    certfn = cert_dir+vidx+'_au.txt'
    start = float(time_dict[uttidx][0]) ### in sec
    veryend = float(time_dict[uttidx][1])
    vlst = []
    print certfn
    if os.path.isfile(certfn):
        for pred_sec in pred_sec_lst:
            end = min(veryend,start+pred_sec)
            featmatlst = cert_feat_mat(certfn,start,end)
            fsize = len(featmatlst)
            vlst.append(featmatlst)
    else:
        for pred_sec in pred_sec_lst:
            vlst.append([0]*fsize)
    totalvlst.append(vlst)
X_v = np.array(totalvlst)
X_v = np.swapaxes(X_v,1,2)

### fix empty
for pred_sec in pred_sec_lst:
    X_bar = np.mean(X_v[:,:,pred_sec-1],axis=0)
    for idx in range(X_v.shape[0]):
        if np.sum(X_v[idx,:,pred_sec-1])==0:
            X_v[idx,:,pred_sec-1] = X_bar
print X_v.shape
pickle.dump(X_v,open('video_seq.pkl','wb'))
"""
### text

aligndir = '../text/align_text/'
totaltlst = []
unigram_idx = pickle.load(open('unigram_idx.pkl'))
duration_dict = pickle.load(open('duration.pkl'))
text_dict = pickle.load(open('text_dict.pkl'))
for line in file(labfn):
    lst = line.split()
    tokenlst,endtimelst = prediction_feature(aligndir+lst[0]+'.txt')
    tlst = []
    for pred_sec in pred_sec_lst:
        feat = [0]*len(unigram_idx)
        ### feature from false alignment
        for token,endtime in zip(tokenlst,endtimelst):
            if endtime<pred_sec:
                #feat[unigram_idx[token]]+=1
                feat[unigram_idx[token]]=1

        ### feature from full observation
        if duration_dict[lst[0]]<pred_sec:
            feat = [0]*len(unigram_idx)
            raw_text = text_dict[lst[0]]
            for token in raw_text.split():
                #feat[unigram_idx[token]]+=1
                feat[unigram_idx[token]]=+1
    
        tlst.append(feat)
    totaltlst.append(tlst)
X_t = np.array(totaltlst)
X_t = np.swapaxes(X_t,1,2)
print X_t.shape
pickle.dump(X_t,open('text_seq.pkl','wb'))
