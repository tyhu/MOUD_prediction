import os
import glob

openSMILE = '/home2/tingyaoh/opensmile-2.0-rc1/opensmile/inst/bin/SMILExtract'
wavlst = glob.glob('../VideoReviews/audioFiles/*.wav')
#conf = 'config/emo_large.conf'
conf = 'config/emobase2010_csv_seq.conf'
pred_sec_lst = range(1,11)

cmd = 'rm opensmile_feat/*_seq.feat'
os.system(cmd)

for pred_sec in pred_sec_lst:
    for fn in wavlst:
        wavid = fn.split('/')[-1].split('.')[0]
        outfn = 'opensmile_feat/'+wavid+'_seq.feat'
        cmd = openSMILE + ' -C '+conf+' -I '+fn+' -start 0 -end '+str(pred_sec)+' -O '+outfn
        print cmd
        os.system(cmd)
