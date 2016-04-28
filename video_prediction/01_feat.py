### extract features from _au files
### Ting-Yao Hu 2016.03

from util_ml import *
import sys
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


cert_dir = '../MultimodalFeatures/visualFeaturesCERT/'
time_lab_fn = 'time_label.txt'
outfn = 'video_feat.txt'
labdict = {'neutral':0,'positive':1,'negative':2}

totalfeat = []
for line in file(time_lab_fn):
    lab = labdict[line.split()[1]]
    start = float(line.split()[2]) ### in sec
    end = float(line.split()[3])
    vidx = line.split()[0].split('_')[0]+'_'+line.split()[0].split('_')[1]
    certfn = cert_dir+vidx+'_au.txt'
    if os.path.isfile(certfn):
        featmatlst = cert_feat_mat(cert_dir+vidx+'_au.txt',start,end)
        totalfeat.append([lab]+featmatlst)
np.savetxt(outfn,np.array(totalfeat),delimiter=',')
