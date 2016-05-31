import sys
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from util_ml import *
from sklearn.metrics import accuracy_score

labfn = '/home2/tingyaoh/sentiment/MOUD/MultimodalFeatures/audioFeatures/cats.txt'

lab = [l.split()[1] for l in file(labfn)]
y = (np.array(lab)=='positive').astype('int')

fnlst = [l.split()[0] for l in file(labfn)]
batchlst = [l.split()[0].split('_')[0] for l in file(labfn)]
xlst = []
for fn in fnlst:
    x = file('opensmile_feat/'+fn+'_pred.feat').readline().split(';')
    xlst.append(x)
X = np.array(xlst).astype('float')
#X = batch_norm(X,np.array(batchlst))
#print np.sum(X[:,0])

np.random.seed(1234)
X, y = RandomPerm(X,y)

ytest_total = []
ypred_total = []
for Xtrain, ytrain, Xtest, ytest in KFold(X,y,5):
    #pca = PCA(n_components=300)
    #pca.fit(Xtrain)
    #Xtrain = pca.transform(Xtrain)
    #Xtest = pca.transform(Xtest)

    #clf = SVC(C=1,kernel='linear')
    #print 'fold'
    #clf = LinearSVC(C=0.01,penalty='l1',dual=False)
    #clf = LinearSVC(C=0.01,penalty='l1',dual=False)
    clf = LogisticRegression(C=0.001)
    #clf = SVC(C=10,gamma=100)
    clf.fit(Xtrain,ytrain)
    ypred = clf.predict(Xtest)
    print accuracy_score(ypred,ytest)
    print ypred
    ytest_total+=ytest.tolist()
    ypred_total+=ypred.tolist()

print accuracy_score(ytest_total,ypred_total)

