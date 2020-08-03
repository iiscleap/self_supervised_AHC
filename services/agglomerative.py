#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:51:43 2019

@author: prachi singh 
@email: prachisingh@iisc.ac.in 
"""

import numpy as np
import argparse
import sys
import os
import matplotlib.pyplot as plt
import pickle
from pdb import set_trace as bp
import subprocess
import scipy.io as sio
from scipy.sparse import coo_matrix

#updating
def setup():
    """Get cmds and setup directories."""
    cmdparser = argparse.ArgumentParser(description='Do speaker clsutering based on'\
                                                    'my ahc',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdparser.add_argument('--threshold', help='threshold for clustering',
                            type=float, default=None)
    cmdparser.add_argument('--lamda', help='lamda for clustering',
                            type=float, default=0.1)
    # cmdparser.add_argument('--custom-dist', help='e.g. euclidean, cosine', type=str, default=None)
    cmdparser.add_argument('--reco2utt', help='spk2utt to create labels', default='../swbd_diar/exp/callhome1/spk2utt')
    cmdparser.add_argument('--reco2num', help='reco2num_spk to get true speakers', default='None')
    cmdparser.add_argument('--label-out', dest='out_file',
                           help='output file used for storing labels', default='../generated_rttm_new/rttm_callhome_my_clustering/cosine/labels')
    # cmdparser.add_argument('--minMaxK', nargs=2, default=[2, 10])
    cmdparser.add_argument('--score_file', help='file containing list of score matrices', type=str,default='../lists/callhome1/callhome1.list')
    cmdparser.add_argument('--score_path', help='path of scores', type=str,default='../scores_cosine/callhome1_scores')
    cmdparser.add_argument('--using_init', help='if initialisation is needed', type=int,default=0)
    cmdparser.add_argument('--dataset', help='dataset name', type=str, default="callhome1")
    # cmdparser.add_argument('--out_path', help='path of output scores', type=str, default=None)

    cmdargs = cmdparser.parse_args()
    return cmdargs

def compute_affinity_matrix(X):
        """Compute the affinity matrix from data.

        Note that the range of affinity is [0,1].

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            affinity: numpy array of shape (n_samples, n_samples)
        """
        # Normalize the data.
        l2_norms = np.linalg.norm(X, axis=1)
        X_normalized = X / l2_norms[:, None]
        # Compute cosine similarities. Range is [-1,1].
        cosine_similarities = np.matmul(X_normalized, np.transpose(X_normalized))
        # Compute the affinity. Range is [0,1].
        # Note that this step is not mentioned in the paper!
        affinity = cosine_similarities

        # affinity = (cosine_similarities + 1.0) / 2.0
        return affinity


def unique(arr, return_ind=False):
    if return_ind:
        k = 0
        d = dict()
        uniques = np.empty(arr.size, dtype=arr.dtype)
        indexes = np.empty(arr.size, dtype='i')
        for i, a in enumerate(arr):
            if a in d:
                indexes[i] = d[a]
            else:
                indexes[i] = k
                uniques[k] = a
                d[a] = k
                k += 1
        return uniques[:k], indexes
    else:
        _, idx = np.unique(arr, return_index=True)
        return arr[np.sort(idx)]


class clustering:
    def __init__(self,n_clusters,clusterlen,lamda,labelfull,dist=None):
        self.n_clusters = n_clusters
        self.labelfull = labelfull.copy()
        self.mergeind = []
        self.eta = 0.1
        self.kc = 2
        self.max_10per_scores = 5
        self.lamda = lamda
        self.clusterlen = clusterlen.copy()
       
        # self.clusterlen=[1]*len(labelfull)
        self.dist = dist
        self.minloss_current = 1000

    def initialize_clusters(self,A):
        sampleNum = len(A)
        NNIndex = np.argsort(A)[:,::-1]
        clusterLabels = np.zeros((sampleNum, 1))
        counter = 0
        for i in range(sampleNum):
            idx = NNIndex[i,:2]
            assignedCluster = clusterLabels[idx]
            assignedCluster = np.unique(assignedCluster[assignedCluster > 0])
            if len(assignedCluster) == 0:
                clusterLabels[idx] = counter
                counter = counter + 1
            elif len(assignedCluster) == 1:
                clusterLabels[idx] = assignedCluster
            else:
                clusterLabels[idx] = assignedCluster[0];            
                for j in range(1,len(assignedCluster)):
                    clusterLabels[clusterLabels == assignedCluster[j]] = assignedCluster[0]
            
        uniqueLabels = np.unique(clusterLabels)
        clusterNumber = len(uniqueLabels)
        
        self.labelfull = clusterLabels[:,0].astype(int)
        initialClusters = []
        output_new = A.copy()
        clusterlist=[]
        for i,lab in enumerate(uniqueLabels):
            ind=np.where(clusterLabels==lab)[0]
            cluster_count = len(ind)
            initialClusters.append(cluster_count)
            clusterlist.append(ind[0])
            avg=np.sum(output_new[ind],axis=0)
            output_new[ind[0]]=avg
            output_new[:,ind[0]]=avg
        #     initialClusters{i} = find(clusterLabels(:) == uniqueLabels(i));
        self.clusterlen = initialClusters
        output_new = output_new[np.ix_(clusterlist,clusterlist)]
        return self.labelfull,self.clusterlen,output_new  

    def compute_distance(self):
        colvec = np.array(self.clusterlen).reshape(-1,1)
        tmp_mat = np.dot(colvec,colvec.T)
        return (1/tmp_mat)

    def Ahc_full(self,A):
        self.A = A.copy()
        while 1:        
            B = self.A.copy()
            tmp_mat=self.compute_distance()
            self.A = self.A*tmp_mat # all elementwise operation
            self.A = np.triu(self.A,k=1)
            cur_samp = self.A.shape[0]
            minA = np.min(self.A)
            self.A[np.tril_indices(cur_samp)]=-abs(minA)*100
            if self.n_clusters != None:
                if cur_samp == self.n_clusters:
                    return self.labelfull,self.clusterlen,self.mergeind
                if self.dist!=None:
                   if ((self.A<self.dist).all() or cur_samp==1):
                        return self.labelfull,self.clusterlen,self.mergeind
            else:
                if (self.A<self.dist).all() or cur_samp==1:
                    return self.labelfull,self.clusterlen,self.mergeind
           
            ind = np.where(self.A==np.amax(self.A))
            minind = min(ind[0][0],ind[1][0])
            maxind = max(ind[0][0],ind[1][0])
            trackind = [list(np.where(self.labelfull==minind)[0])]
            trackind.extend(np.where(self.labelfull==maxind)[0])
            if minind == maxind:
                print(minind,maxind)
            self.clusterlen[minind] +=self.clusterlen[maxind]
            self.clusterlen.pop(maxind)
            self.labelfull[np.where(self.labelfull==maxind)[0]]=minind
            unifull = list(np.unique(self.labelfull))
            labelfullnew = np.zeros(self.labelfull.shape).astype(int)
            for i in range(len(self.labelfull)):
                labelfullnew[i]=unifull.index(self.labelfull[i])
            self.labelfull = labelfullnew
            self.mergeind.append(trackind)
            newsamp = cur_samp -1
            # recomputation
            B[:,minind] =B[:,minind]+B[:,maxind]
            B[minind] = B[:,minind]
            B = np.delete(B,maxind,1)
            B = np.delete(B,maxind,0)
            B[np.diag_indices(newsamp)]=np.min(B)
            B[np.diag_indices(newsamp)] = np.max(B,axis=1)
            self.A = B.copy()
        return self.labelfull,self.clusterlen,self.mergeind


    def my_clustering_full(self,A):

        if self.lamda == 0.0 :
            return self.Ahc_full(A)
        self.A = A.copy()
        while 1:
            B = self.A.copy()
            tmp_mat=self.compute_distance()
            #bp()
            self.A = self.A*tmp_mat # all elementwise operation
            self.A = np.triu(self.A,k=1)
            cur_samp = self.A.shape[0]
            minA=np.min(self.A)
            self.A[np.tril_indices(cur_samp)]=-abs(minA)*100
            # if cur_samp<=4:
            #     bp()
            if self.n_clusters != None:
                if cur_samp == self.n_clusters:
                    return self.labelfull,self.clusterlen,self.mergeind
                if self.dist!=None:
                   if ((self.A<self.dist).all() or cur_samp==1):
                        return self.labelfull,self.clusterlen,self.mergeind
            else:
                if (self.A<self.dist).all() or cur_samp==1:
                    return self.labelfull,self.clusterlen,self.mergeind

            # self.kc = max(1,int(np.ceil(self.eta*cur_samp)))
            self.kc = min(self.kc,cur_samp)
            # self.max_10per_scores = max(1,int(self.eta*cur_samp))
            self.max_10per_scores = min(self.max_10per_scores,cur_samp)
            # max_10per_scores = 10
            ind = np.argsort(self.A,axis=1)
            myind = (np.arange(self.A.shape[0]),ind[:,-1])
            rowmaxind = np.argsort(self.A[myind],axis=None)[::-1][:self.max_10per_scores]
            minloss = 1000
            for row in rowmaxind:
                # cind=ind[row][-self.kc:]
                cindall = ind[row]
                cindall = cindall[cindall>row]
                cindall = cindall[::-1]
                for j in range(len(cindall)):
                    maxcind=cindall[j]
                    cind = np.delete(cindall,j,0)
                    # if A[row,maxcind] == 0.0:
                    #     continue
                    avg_link = 0
                    cluster_sum = B[row,cind] + B[maxcind,cind]
                    cluster_length = (self.clusterlen[row]+self.clusterlen[maxcind])*np.array(self.clusterlen)[cind]
                    cluster_scores = cluster_sum/cluster_length
                    cluster_scores = np.sort(cluster_scores)[::-1][:self.kc-1]
                    avg_link = np.sum(cluster_scores)
                    loss = - self.A[row,maxcind] + self.lamda*avg_link
                    # loss = np.round(loss,3)
                    if loss < minloss:
                                            
                        minloss = loss
                        
                        rowmax = row
                        colmax = maxcind
            # print('init_clusters:{} final_clusters:{} diff_of_ahc_loss:{}'.format(len(A),self.n_clusters,minloss-self.minloss_current))
            self.minloss_current = minloss
            # mergeind.append([rowmax,colmax])
            try:
                minind=min(colmax,rowmax)
            except:
                print('cur_clusters:',cur_samp,'needed_clusters:',self.n_clusters)
                raise ValueError('Not enough clusters!')
                
            maxind=max(colmax,rowmax)
            trackind = [list(np.where(self.labelfull==minind)[0])]
            trackind.extend(np.where(self.labelfull==maxind)[0])
            if minind == maxind:
                print(minind,maxind)
            self.clusterlen[minind] +=self.clusterlen[maxind]
            self.clusterlen.pop(maxind)
            self.labelfull[np.where(self.labelfull==maxind)[0]]=rowmax
            unifull = list(np.unique(self.labelfull))
            labelfullnew = np.zeros(self.labelfull.shape).astype(int)
            for i in range(len(self.labelfull)):
                labelfullnew[i]=unifull.index(self.labelfull[i])
            # if len(self.clusterlen)==22:
            #     bp()
            # print(len(self.clusterlen),len(unifull))
            self.labelfull = labelfullnew
            self.mergeind.append(trackind)
            newsamp = cur_samp-1
            
            # recomputation
            B[:,minind] =B[:,minind]+B[:,maxind]
            B[minind] = B[:,minind]
            B = np.delete(B,maxind,1)
            B = np.delete(B,maxind,0)
            B[np.diag_indices(newsamp)]=np.min(B)
            B[np.diag_indices(newsamp)] = np.max(B,axis=1)
            self.A = B.copy()
        return self.labelfull,self.clusterlen,self.mergeind

    def get_params(self):
        return self.labelfull, self.mergeind


def write_results_dict(results_dict, output_file,reco2utt):
    """Writes the results in label file"""

    output_label = open(output_file,'w')
    reco2utt = open(reco2utt,'r').readlines()
    i=0

    for meeting_name, hypothesis in results_dict.items():
        
        reco = reco2utt[i].split()[0]
        utts = reco2utt[i].rstrip().split()[1:]
        if reco == meeting_name:
            for j,utt in enumerate(utts):
                towrite = utt +' '+str(hypothesis[j])+'\n'
                output_label.writelines(towrite)     
        else:
            print('reco mismatch!')
            
             
        i=i+1


def callhome_clustering():
    args = setup()
    fold = args.score_path
    file_list = np.genfromtxt(args.score_file,dtype=str)
    out_file = args.out_file
    reco2utt = args.reco2utt
    reco2num = args.reco2num
    threshold=args.threshold
    lamda = args.lamda
    dataset = fold.split('/')[-3]
    prior = 0
    alpha = 0.5

    print(threshold)
    if reco2num != 'None':
        reco2num_spk = open(args.reco2num).readlines()
    results_dict ={}
    if "callhome1" in fold:
        dataset="callhome1_scores"
    else:
        dataset="callhome2_scores"
    for i,f in enumerate(file_list):
        print(f)
        
        if not args.using_init:
            if "baseline" in fold:
                b = np.load(fold+'/'+f+'.npy')
            else:
                deepahcmodel = pickle.load(open(fold+'/'+f+'.pkl','rb'))
                b = deepahcmodel['output']
                # averaging with PLDA
                # b_plda = np.load("../scores_plda_new/{}/{}.npy".format(dataset,f))
                # b = alpha*b + (1-alpha)*b_plda
            clusterlen = [1]*b.shape[0]
            labels = np.arange(b.shape[0])
        else:
            deepahcmodel = pickle.load(open(fold+'/'+f+'.pkl','rb'))
            b = deepahcmodel['reduced_output']
            clusterlen = deepahcmodel['clusterlen']
            labels = deepahcmodel['labelfull']
        if reco2num != 'None':
            n_clusters = int(reco2num_spk[i].split()[1])      
            n_clusters = min(n_clusters,len(clusterlen)) # whichever is minimum
            if f!=reco2num_spk[i].split()[0]:
                print('file mismatch',f,reco2num_spk[i].split()[0])
            threshold = None
        else:
            n_clusters = None         
            clus =clustering(n_clusters,clusterlen,lamda,labels,dist=threshold)
        labelfull,_,mergeind=clus.my_clustering_full(b)
        uni1,method1=unique(labelfull,True)
        results_dict[f]=method1     

    write_results_dict(results_dict, out_file,reco2utt)


if __name__ == "__main__":
    args = setup()
    fold = args.score_path
    callhome_clustering()
    

