import os
import glob

openSMILE = '/home2/tingyaoh/opensmile-2.0-rc1/opensmile/inst/bin/SMILExtract'
wavlst = glob.glob('../VideoReviews/audioFiles/*.wav')
#conf = 'config/emo_large.conf'
conf = 'config/emobase2010_csv.conf'

for fn in wavlst:
    wavid = fn.split('/')[-1].split('.')[0]
    outfn = 'opensmile_feat/'+wavid+'.feat'
    cmd = openSMILE + ' -C '+conf+' -I '+fn+' -O '+outfn
    print cmd
    os.system(cmd)
