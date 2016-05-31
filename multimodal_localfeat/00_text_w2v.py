### Ting-Yao Hu 2016.05

import sys
from myconfig import *
sys.path.append(util_dir)
import glob, string
import cPickle as pickle
from util_ml import *
import gensim, logging

print '### loading w2v model'
model = gensim.models.Word2Vec.load_word2vec_format(w2v_dir+'SBW-vectors-300-min5.bin', binary=True)

text_dict = {}
duration_dict = {}
text_dir = '../VideoReviews/transcriptions/'
textfnlst = glob.glob(text_dir+'*.csv')
for fn in textfnlst:
    infile = file(fn)
    infile.readline()
    count=0
    key = fn.split('/')[-1].split('.')[0]
    for line in infile:
        count+=1
        key_utt = key+'_'+str(count)
        dur = float(line.split(';')[1])-float(line.split(';')[0])
        duration_dict[key_utt] = dur
        raw_text = line.split(';')[2][1:-1]
        raw_text = "".join([ch for ch in raw_text if ch not in string.punctuation])
        raw_text = raw_text.lower()
        text_dict[key_utt] = raw_text
pickle.dump(duration_dict,open('duration.pkl','wb'))
pickle.dump(text_dict,open('text_dict.pkl','wb'))

labfn = '../MultimodalFeatures/audioFeatures/cats.txt'
featlst,lablst = [],[]
labdict = {'neutral':0,'positive':1,'negative':2}
for line in file(labfn):
    lst = line.split()
    raw_text = text_dict[lst[0]]
    uttfeatlst = []
    for token in raw_text.split():
        try:
            uttfeatlst.append(model[token].tolist())
        except KeyError:
            print 'token: '+token+' as OOV'
            uttfeatlst.append([0]*300)
    uttfeat = np.array(uttfeatlst)
    featlst.append(uttfeat)
pickle.dump(featlst,open('text_w2v.pkl','wb'))
