#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:39:25 2020

@author: prachi
"""

import os
import sys
import numpy as np
import subprocess
import pickle
import argparse


#updating
def setup():
    """Get cmds and setup directories."""
    cmdparser = argparse.ArgumentParser(description='convert kaldi PCA transform and mean into pickle format',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cmdparser.add_argument('--kaldi_feats_path', help='path of folder where transform.mat and mean.vec stored', type=str,required=True)
    cmdparser.add_argument('--dataset', help='dataset name', type=str, default="callhome1",required=True)


    cmdargs = cmdparser.parse_args()
    return cmdargs

def kaldiPlda2numpydict(pldaFile):
    #logging.debug('kaldi text file to numpy array: {}'.format(textfile))
    fin = subprocess.check_output(["ivector-copy-plda", "--binary=false", pldaFile ,"-"])
    res = {}
    fin = fin.decode("utf-8").split('\n')
    while '' in fin:
        fin.remove('')
    splitted = fin[0].strip().split()
    res['plda_mean'] = np.asarray(splitted[2:-1]).astype(float)
    tmparr=[]
    for i,line in enumerate(fin[2:]):
        splitted = line.strip().split()
        if splitted[-1] == ']':
            splitted = splitted[:-1]
            tmparr.append(np.asarray(splitted).astype(float))
            break
        else:
            tmparr.append(np.asarray(splitted).astype(float))
    res['diagonalizing_transform'] = np.asarray(tmparr)
    res['Psi_across_covar_diag'] = np.asarray(fin[-2].strip().split()[1:-1]).astype(float)
    
    return res
     
def load_kaldi_matrices(args):
    fold_local = args.kaldi_feats_path
    dataset = args.dataset
    outpicklefile = 'lists/{}/plda_{}.pkl'.format(dataset,dataset)

    if os.path.isfile(outpicklefile):
        print("file exits!")
        return
        
    plda_file = '{}/plda'.format(fold_local)
    if os.path.isfile(plda_file):
        plda = kaldiPlda2numpydict(plda_file)
    else:
        print('plda model does not exist!')
        plda = {}
    transform_mat_file = '{}/transform.mat'.format(fold_local)
    mean_vec_file = '{}/mean.vec'.format(fold_local)
    transform_mat = np.asarray([w.split() for w in np.asarray(subprocess.check_output(["copy-matrix","--binary=false", transform_mat_file, "-"]).decode('utf-8').strip()[2:-2].split('\n'))]).astype(float)
    mean_vec = np.asarray(subprocess.check_output(["copy-vector", "--binary=false",mean_vec_file, "-"]).decode('utf-8').strip()[1:-2].split()).astype(float)
    plda['transform_mat'] = transform_mat
    plda['mean_vec'] = mean_vec
    
    with open(outpicklefile,'wb') as f:
            pickle.dump(plda,f)

if __name__=='__main__':
    args = setup()
    load_kaldi_matrices(args)
        
        
    
    
    
    
    
    