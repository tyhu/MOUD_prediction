### Ting-Yao Hu, 2016.05

import sys
import os
from util_ml import *

labfn = '/home2/tingyaoh/sentiment/MOUD/MultimodalFeatures/audioFeatures/cats.txt'
lab = [l.split()[1] for l in file(labfn)]
vidx = [l.split()[0] for l in file(labfn)]
y = (np.array(lab)=='positive').astype('int')
vidx = np.array(vidx)

np.random.seed(1234)
vidx, y = RandomPerm(vidx,y)
for vid in vidx: print vid
