### Ting-Yao Hu, 2016.04
### use sphinx4 to do force alignment

import sys
import os
import glob
import string

sphinx4jar='/home/ubuntu/cmusphinx/sphinx4-5prealpha-src/sphinx4-core/target/sphinx4-core-1.0-SNAPSHOT.jar'
AM='/home/ubuntu/cmusphinx/voxforge-es-0.2/model_parameters/voxforge_es_sphinx.cd_ptm_3000'
DICTIONARY='/home/ubuntu/cmusphinx/voxforge-es-0.2/etc/voxforge_es_sphinx.dic'

aligndir = 'align_text/'
textdir = '../VideoReviews/transcriptions/'
audiodir = '../VideoReviews/audioFiles/'
textfnlst = glob.glob(textdir+'*.csv')
for fn in textfnlst:
    vid = fn.split('/')[-1].split('.')[0]
    count = 0
    for line in file(fn,'r'):
        if line[1]=='#': continue
        count+=1
        audiofn = audiodir+vid+'_'+str(count)+'.wav'
        raw_text = line.split(';')[2][1:-1]
        raw_text = "".join([ch for ch in raw_text if ch not in string.punctuation])
        raw_text = raw_text.lower()
	print vid+'_'+str(count)
        cmd = 'sox '+audiofn+' -b 16 sample.wav channels 1 rate 16k'
        #print raw_text
        os.system(cmd)
        cmd = 'java -cp '+sphinx4jar+' edu.cmu.sphinx.tools.aligner.Aligner '+AM+' '+DICTIONARY+' sample.wav '+'\"'+raw_text+'\" > '+aligndir+vid+'_'+str(count)+'.txt'
        os.system(cmd)
        #print file('tmp').readline()
