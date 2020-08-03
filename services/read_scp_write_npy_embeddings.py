import os
import numpy as np
import glob
import kaldi_io
import sys


#Usage
if len(sys.argv)!=4:
    print("Need 3 input arguments!")
    print("Usage :")
    print("python read_scp_write_npy_embeddings.py <mat/vec> <complete path of ark/scp file> <path of output folder to store numpy output>")
    print("<mat/vec> : mat if scp contains matrix , vec if scp contains vector e.g. x-vectors")


arkscppath = sys.argv[2] 
outputnpyfilepath = sys.argv[3] 
if not os.path.isdir(outputnpyfilepath):
	print('Creating directory where npy scores will be saved : {}'.format(outputnpyfilepath))
	os.makedirs(outputnpyfilepath)
else:
    print("xvectors numpy path exists !")
    exit()

ext = os.path.splitext(file_name)[1]
if sys.argv[1]=='mat':
    #for score files
    if ext==".scp":
	   d = { key:mat for key,mat in kaldi_io.read_mat_scp(arkscppath) }
    else:
        print("File type not correct. scp required.")
elif sys.argv[1]=='vec':
	#for embeddings
    if ext==".scp":
	   d = { key:mat for key,mat in kaldi_io.read_vec_flt_scp(arkscppath) }
    elif ext == ".ark":
        d = { key:mat for key,mat in kaldi_io.read_vec_flt_ark(arkscppath) }
    else:
        print("File type not correct. scp/ark required.")
else:
    print("first argument should be mat/vec ")




for count,(i,j) in enumerate(d.items()):
	if count % 100 == 0:
		print("Done with {} files".format(count))
	np.save(outputnpyfilepath+'/'+i,j)
