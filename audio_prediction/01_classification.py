import sys

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from util_ml import *
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

labfn = '../MultimodalFeatures/audioFeatures/cats.txt'

lab = [l.split()[1] for l in file(labfn)]
y = (np.array(lab)=='positive').astype('int')

fnlst = [l.split()[0] for l in file(labfn)]
batchlst = [l.split()[0].split('_')[0] for l in file(labfn)]
xlst = []
for fn in fnlst:
    x = file('opensmile_feat/'+fn+'.feat').readline().split(';')
    xlst.append(x)
X = np.array(xlst).astype('float')
#X = batch_norm(X,np.array(batchlst))
#print np.sum(X[:,0])

np.random.seed(1234)
#np.random.seed(1238)
X, y = RandomPerm(X,y)

ytest_total = []
ypred_total = []
for Xtrain, ytrain, Xtest, ytest in KFold(X,y,5):
    print np.sum(ytrain==1)
    print np.sum(ytrain==0)
    #fs = SelectKBest(f_classif, k=325)
    #fs.fit(Xtrain, ytrain)
    #Xtrain = fs.transform(Xtrain)
    #Xtest = fs.transform(Xtest)

    #clf = SVC(C=1,kernel='linear')
    #print 'fold'
    #clf = LinearSVC(C=1)
    #clf = LinearSVC(C=1,penalty='l1',dual=False)
    #clf = LogisticRegression(C=1,penalty='l1')
    clf = LogisticRegression(C=0.001) ### current best
    #clf = LogisticRegression(C=1,penalty='l1')
    #clf = SVC(C=10,gamma=0.01)
    clf.fit(Xtrain,ytrain)
    ypred = clf.predict(Xtest)
    ypred_tr = clf.predict(Xtrain)
    print 'train acc: ',accuracy_score(ypred_tr,ytrain)
    print accuracy_score(ypred,ytest)
    ytest_total+=ytest.tolist()
    ypred_total+=ypred.tolist()

print accuracy_score(ytest_total,ypred_total)

