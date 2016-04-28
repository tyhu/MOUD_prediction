### extract time stamp for each utterance
### Ting-Yao Hu 2016.03

import sys

labfn = '../VideoReviews/cats.txt'
#labfn = '../MultimodalFeatures/audioFeatures/cats.txt'
trans_dir = '../VideoReviews/transcriptions/'
outfn = 'video_time_stamp.txt'
outfile = file(outfn,'w')
lastvidx = ''
for line in file(labfn):
    vidx = line.split()[0].split('_')[0]+'_'+line.split()[0].split('_')[1]
    if vidx!=lastvidx:
        tfile = file(trans_dir+vidx+'.csv')
        tfile.readline()
        count = 0
        for l in tfile:
            lst  = l.split(';')
            count+=1
            outln = vidx+'_'+str(count)+' '+lst[0]+' '+lst[1]
            outfile.write(outln+'\n')
    lastvidx = vidx
outfile.close()

### combine
labfile = file(labfn)
timefile = file(outfn)
outfile = file('time_label.txt','w')
while True:
    l = labfile.readline().strip()
    if len(l)==0: break
    tlst = timefile.readline().split()
    outl = l+' '+tlst[1]+' '+tlst[2]
    outfile.write(outl+'\n')
