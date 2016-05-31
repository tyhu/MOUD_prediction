### rnn
### Ting-Yao Hu, 2016.05

import sys
import os
from myconfig import *
sys.path.append(util_dir)
sys.path.append(keras_wrap_dir)
from util_ml import *
from keras_rnn import *
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

X_vs = pickle.load(open(feat_dir+'video_seq.pkl'))
X_as = pickle.load(open(feat_dir+'audio_seq.pkl'))
y = pickle.load(open(feat_dir+'lab.pkl'))
y_dummy = pickle.load(open(feat_dir+'lab.pkl'))
l = pickle.load(open(feat_dir+'length.pkl'))

datanum = X_vs.shape[0]
featnum = X_vs.shape[1]
np.random.seed(1234)
X_vs, y = RandomPerm(X_vs,y)
np.random.seed(1234)
l, y_dummy = RandomPerm(l,y_dummy)
np.random.seed(1234)
X_as, y_dummy = RandomPerm(X_as,y_dummy)

for idx in range(10):
    X_as[:,:,idx] = preprocessing.scale(X_as[:,:,idx])

#keras_rnn = KerasRNN(featnum, 30, 2, ld=0.0001) #0.728, video only
keras_rnn = KerasRNN(featnum, 20, 2, ld=0.001, rtype='lstm')
#keras_rnn = KerasRNN(featnum, 5, 2, ld=100, rtype='lstm') 
keras_rnn.compile()
model = keras_rnn.model

X_avs = X_vs
ytest_total = []
ypred_total = []
maxl = 10
for Xtrain, ytrain, ltrain, Xtest, ytest, ltest in KFold_withl(X_avs,y,l,5):
    Xtrain = np.swapaxes(Xtrain,1,2)
    ytr = to_categorical(np.array(ytrain)-1,2)
    for it in range(200):
        for idx in range(Xtrain.shape[0]):
            ll = min(int(np.ceil(ltrain[idx])),maxl)
            X_s = Xtrain[idx:idx+1,:ll,:]
            y_s = ytr[idx:idx+1,:]
            #print X_s.shape
            #print y_s.shape
            #model.fit(X_s,y_s,nb_epoch=1)
            model.train_on_batch(X_s,y_s)
    
    Xtest = np.swapaxes(Xtest,1,2)
    for idx in range(Xtest.shape[0]):
        ll = min(int(np.ceil(ltest[idx])),maxl)
        pred = model.predict(Xtest[idx:idx+1,:ll,:])
        pred = np.argmax(pred[0])+1
        ypred_total.append(pred)
    ytest_total+=ytest.tolist()

print accuracy_score(ytest_total,ypred_total)
