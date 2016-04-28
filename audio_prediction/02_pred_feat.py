import os
import glob

openSMILE = '/home2/tingyaoh/opensmile-2.0-rc1/opensmile/inst/bin/SMILExtract'
wavlst = glob.glob('../VideoReviews/audioFiles/*.wav')
#conf = 'config/emo_large.conf'
conf = 'config/emobase2010_csv_seg.conf'
pred_sec = 2

for fn in wavlst:
    wavid = fn.split('/')[-1].split('.')[0]
    outfn = 'opensmile_feat/'+wavid+'_pred.feat'
    cmd = openSMILE + ' -C '+conf+' -I '+fn+' -start 0 -end '+str(pred_sec)+' -O '+outfn
    print cmd
    os.system(cmd)
