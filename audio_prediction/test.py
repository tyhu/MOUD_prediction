
from sklearn.svm import SVC
from util_ml import *
from sklearn.metrics import accuracy_score

inputfn = '/home2/tingyaoh/sentiment/MOUD/MultimodalFeatures/audioFeatures/spanish.AudioFeatures.csv'

data = np.genfromtxt(inputfn,delimiter=',',skip_header=1,dtype='str')
data = data[:,1:]

lab = data[:,-1]
X = data[:,:-1].astype('float')
y = lab=='positive'

np.random.seed(1234)
X, y = RandomPerm(X,y)
ytest_total = []
ypred_total = []
for Xtrain, ytrain, Xtest, ytest in KFold(X,y,10):
    clf = SVC(C=0.1,kernel='linear')
    clf.fit(Xtrain,ytrain)
    ypred = clf.predict(Xtest)
    ytest_total+=ytest.tolist()
    ypred_total+=ypred.tolist()

print accuracy_score(ytest_total,ypred_total)
