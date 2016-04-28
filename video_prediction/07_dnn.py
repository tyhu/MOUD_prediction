### Ting-Yao Hu 2016.03

import sys
import keras
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
sys.path.append('/home2/tingyaoh/ML/PyAI/deep_learn_keras/')
from keras_mlp import *
from sklearn.metrics import accuracy_score
from util_ml import *

labfn = '../VideoReviews/cats.txt'
lab = [l.split()[1] for l in file(labfn)]
y = np.array(lab)

featfn = 'video_feat.txt'
feat = np.genfromtxt(featfn,delimiter=',',skip_header=0).astype('float')
y = np.array(feat[:,0].tolist())

y = y[np.where(y!=0)]
X = feat[:,1:]
X = X[np.where(y!=0)]
X = np.nan_to_num(X)

print X.shape,y.shape

X_bar = np.mean(X,axis=0)
for idx in range(X.shape[0]):
    if np.sum(X[idx])==0: X[idx] = X_bar

#np.random.seed(1234)
np.random.seed(1234)
X, y = RandomPerm(X,y)
ytest_total = []
ypred_total = []

for Xtrain, ytrain, Xtest, ytest in KFold(X,y,5):

    mlp = KerasMLP([Xtrain.shape[1],10,2],['tanh','tanh','softmax'],droprate=0,ld=0.001)
    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
    mlp.compile(optimizer=sgd)

    ytrain_nn,cat = NNFormLabel(ytrain)
    mlp.model.fit(Xtrain,ytrain_nn,nb_epoch=20000, batch_size=16,show_accuracy=True)
    ypred = mlp.model.predict_classes(Xtest)
    ytest_nn, _ = NNFormLabel(ytest,cat)
    
    ytest = ytest_nn[:,1]
    #ypred = NNLabel2NormLabel(ypred,cat)
    print accuracy_score(ypred,ytest)
    ytest_total+=ytest.tolist()
    ypred_total+=ypred.tolist()

print accuracy_score(ytest_total,ypred_total)

