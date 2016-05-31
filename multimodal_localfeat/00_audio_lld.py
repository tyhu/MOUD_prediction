### Ting-Yao Hu 2016.05

import sys
from myconfig import *
sys.path.append(util_dir)
from util_ml import *
import os
import glob
import cPickle as pickle
"""
cmd = 'mkdir opensmile_feat'
os.system(cmd)
conf = 'config/emobase2010_lld.conf'
wavlst = glob.glob('../VideoReviews/audioFiles/*.wav')
for fn in wavlst:
    wavid = fn.split('/')[-1].split('.')[0]
    outfn = 'opensmile_feat/'+wavid+'.feat'
    cmd = openSMILE + ' -C '+conf+' -I '+fn+' -O '+outfn
    print cmd
    os.system(cmd)
"""
labfn = '../MultimodalFeatures/audioFeatures/cats.txt'
featlst,lablst = [],[]
labdict = {'neutral':0,'positive':1,'negative':2}
for line in file(labfn):
    lst = line.split()
    feat = np.genfromtxt('opensmile_feat/'+lst[0]+'.feat',delimiter=';')
    featlst.append(feat[0::5,:])
pickle.dump(featlst,open('audio_lld.pkl','wb'))
