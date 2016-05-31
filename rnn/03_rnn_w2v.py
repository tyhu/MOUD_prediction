### rnn for w2v feature
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

X_t = pickle.load(open(local_feat_dir+'text_w2v.pkl'))
y = pickle.load(open(feat_dir+'lab.pkl'))
y_dummy = pickle.load(open(feat_dir+'lab.pkl'))

np.random.seed(1234)
X,y = RandomPerm(np.array(X_t),y)

keras_rnn = KerasRNN(300, 30, 2, ld=0.001, rtype='lstm')
keras_rnn.compile()
model = keras_rnn.model

ypred_total, ytest_total = [],[]
for Xtrain, ytrain, Xtest, ytest in KFold(X,y,5):
    ytr = to_categorical(np.array(ytrain)-1,2)
    for it in range(200):
        for idx in range(Xtrain.shape[0]):
            X_s = Xtrain[idx][np.newaxis,:,:]
            y_s = ytr[idx:idx+1,:]
            model.train_on_batch(X_s,y_s)

    for idx in range(Xtest.shape[0]):
        pred = model.predict(Xtest[idx][np.newaxis,:,:])
        pred = np.argmax(pred[0])+1
        ypred_total.append(pred)
    ytest_total+=ytest.tolist()

print accuracy_score(ytest_total,ypred_total)
