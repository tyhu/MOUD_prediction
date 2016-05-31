### Ting-Yao Hu, 2016.04

import sys
import os
from sklearn import preprocessing
from sklearn.datasets import dump_svmlight_file
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
sys.path.append('/home2/tingyaoh/ML/PyAI/util')
from util_ml import *

mkltrain = '/home2/tingyaoh/ML/spg-gmkl/svm-train'
mklpredict = '/home2/tingyaoh/ML/spg-gmkl/svm-predict'

X_t = pickle.load(open('text.pkl'))
X_a = pickle.load(open('audio.pkl'))
X_v = pickle.load(open('video.pkl'))
y = pickle.load(open('lab.pkl'))

#X_t = preprocessing.scale(X_t)
X_a = preprocessing.scale(X_a)
X_v = preprocessing.scale(X_v)
X = np.c_[X_t,X_a,X_v]
#X = np.c_[X_a,X_v]
#X = X_a
print X.shape

np.random.seed(1234)
X, y = RandomPerm(X,y)

ytest_total = []
ypred_total = []
for Xtrain, ytrain, Xtest, ytest in KFold(X,y,5):
    #clf = LogisticRegression(C=10000,penalty='l1')
    clf = LogisticRegression(C=100000,penalty='l1')
    #clf = LinearSVC(C=100,penalty='l1',dual=False)
    #clf = LinearSVC(C=1,dual=False)
    clf.fit(Xtrain,ytrain)
    coef = np.abs(clf.coef_[0])
    featlst = range(len(coef))
    featlst = sorted(featlst, key=lambda k:coef[k], reverse=True)
    #print featlst

    #fs = SelectKBest(f_classif, k=10)
    #fs.fit(Xtrain,ytrain)
    

    """
    dump_svmlight_file(Xtrain,ytrain,'train.dat')
    dump_svmlight_file(Xtest,ytest,'test.dat')

    cmd = mkltrain+' -a 2 -g 0 -e 1e-12 -k kernel.txt -c 10 train.dat mkl.mdl'
    print cmd
    os.system(cmd)
    cmd = mklpredict+' test.dat mkl.mdl output.txt'
    print cmd
    os.system(cmd)

    ypred = [int(l.strip()) for l in file('output.txt')]
    """

    ypred = clf.predict(Xtest)
    print accuracy_score(ypred,ytest)
    ytest_total+=ytest.tolist()
    ypred_total+=ypred.tolist()

print accuracy_score(ytest_total,ypred_total)
