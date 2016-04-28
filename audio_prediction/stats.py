
import sys
from util_ml import *

labfn = '/home2/tingyaoh/sentiment/MOUD/MultimodalFeatures/audioFeatures/cats.txt'

lab = [l.split()[1] for l in file(labfn)]
y = (np.array(lab)=='positive').astype('int')

fnlst = [l.split()[0] for l in file(labfn)]
batchlst = [l.split()[0].split('_')[0] for l in file(labfn)]
tlst = []
for fn in fnlst:
    x = file('opensmile_feat/'+fn+'.feat').readline().split(';')
    tlst.append(x[-1])
X = np.array(tlst).astype('float')
print np.sum(X)
print np.mean(X)
print np.std(X)
#X = batch_norm(X,np.array(batchlst))
