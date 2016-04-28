### Ting-Yao Hu 2016.03

import sys
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
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

#print X.shape,y.shape

X_bar = np.mean(X,axis=0)
for idx in range(X.shape[0]):
    if np.sum(X[idx])==0: X[idx] = X_bar

#X,_,_ = normalization(X)

np.random.seed(1234)
#np.random.seed(1234)
X, y = RandomPerm(X,y)
ytest_total = []
ypred_total = []

for Xtrain, ytrain, Xtest, ytest in KFold(X,y,10):
    #clf = LinearSVC(C=1,penalty='l1',dual=False)
    #clf = LogisticRegression(C=10,penalty='l1')
    #clf = LogisticRegression(C=0.1)
    #clf = SVC(gamma=0.001,C=10)  ### current best, with featmean, no normalization
    clf = SVC(gamma=0.001,C=10)
    clf.fit(Xtrain,ytrain)
    ypred = clf.predict(Xtest)
    print ypred
    print accuracy_score(ypred,ytest)
    ytest_total+=ytest.tolist()
    ypred_total+=ypred.tolist()

print accuracy_score(ytest_total,ypred_total)

