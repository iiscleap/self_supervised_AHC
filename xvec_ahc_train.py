#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:51:43 2019

@author: prachi singh 
@email: prachisingh@iisc.ac.in 

This code is for DNN training 
Explained in paper:
P. Singh, S. Ganapathy, Deep Self-Supervised Hierarchical Clustering for Speaker Diarization, Interspeech, 2020

Check main function: train_with_threshold , to run for different iterations
"""

import os
import sys
import numpy as np
import random
import pickle
import subprocess

from collections import OrderedDict
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import torch
import torch.nn as nn
import torch.optim as optim
from models_train_ahc import weight_initialization,Deep_Ahc_model
import torch.utils.data as dloader
from arguments import read_arguments as params
from pdb import set_trace as bp
import services.agglomerative as ahc
sys.path.insert(0,'tools_diar/steps/libs')

# read arguments
opt = params()
#select device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpuid

# torch.manual_seed(777)  # reproducibility



lamda = opt.lamda
loss_lamda = opt.gamma
dataset=opt.dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)

# Model defined here
def normalize(system):
     # to make zero mean and unit variance
        my_mean = np.mean(system)
        my_std = np.std(system)
        system = system-my_mean
        system /= my_std
        return system

def compute_affinity_loss(output,cluster,lamda):

    mylist = np.arange(len(cluster))
    # print('mylist:',mylist)
    loss=0.0
    biglamda=0.0
    for k,c in enumerate(cluster):
        for i in range(len(c)-1):
            for j in range(i+1,len(c)):
                nlist=np.delete(mylist,k,0)
                # bp()
                try:
                    ind=np.random.choice(nlist)
                except:
                    bp()
                b = cluster[ind]
                # bp()
                bi = i % len(b)
                # min(i,abs(i-len(b)))
                loss += -output[c[i],c[j]]+ lamda*output[c[i],b[bi]]+ lamda*output[c[j],b[bi]]
                biglamda +=1

    return loss/biglamda
def mostFrequent(arr, n):

    # Insert all elements in Hash.
    Hash = dict()
    for i in range(n):
        if arr[i] in Hash.keys():
            Hash[arr[i]] += 1
        else:
            Hash[arr[i]] = 1

    # find the max frequency
    max_count = 0
    res = -1
    for i in Hash:
        if (max_count < Hash[i]):
            res = i
            max_count = Hash[i]

    return res

def check_clusterpurity(f,cluster,dataset,gnd_label,sys_label):
    """
    check cluster purity with respect to xvectors belonging to single speakers
    """
    clusterpurity=[]
    clustervar = []
    fullclasslabel = []
    clean_ind =[]
    for c in cluster:
        classlabel=[]
        clustervar.append(np.var(c))
        for a in c:
            if len(gnd_label[a])==1:
                classlabel.append(gnd_label[a][0])
                clean_ind.append(a)
            # else:
            #     sys_label = np.delete(sys_label,a)
        fullclasslabel.extend(classlabel)
        classlabel = np.array(classlabel)

        if len(classlabel)==0:
            clusterpurity.append(0)
            continue
        unilabel = mostFrequent(classlabel,len(classlabel))
        purity = (len(np.where(classlabel==unilabel)[0])/len(classlabel))*100
        
        clusterpurity.append(purity)
    sys_label = sys_label[clean_ind]
    Nmi_score = nmi(fullclasslabel,sys_label.tolist())
    print('NMI score for n_cluster:{} is {}'.format(len(cluster),Nmi_score))
    print('cluster purity for n_cluster:{} is {}'.format(len(cluster),clusterpurity))



class Deep_AHC:
    def __init__(self,pldamodel,fname,reco2utt,xvecdimension,model,optimizer,n_prime,writer=None):
        self.reco2utt = reco2utt
        self.xvecdimension = xvecdimension
        self.model = model
        self.optimizer = optimizer
        self.n_prime = n_prime
        self.fname = fname
        self.final =0
        self.forcing_label = 0
        self.results_dict={}
        self.pldamodel = pldamodel
        self.lamda = lamda
       
       

    def write_results_dict(self, output_file):
        """Writes the results in label file"""
        f = self.fname
        output_label = open(output_file+'/'+f+'.labels','w')

        hypothesis = self.results_dict[f]
        meeting_name = f
        reco = self.reco2utt.split()[0]
        utts = self.reco2utt.rstrip().split()[1:]
        if reco == meeting_name:
            for j,utt in enumerate(utts):
                towrite = utt +' '+str(hypothesis[j])+'\n'
                output_label.writelines(towrite)
        output_label.close()

        rttm_channel=0
        segmentsfile = opt.segments+'/'+f+'.segments'
        python = opt.which_python
      
        cmd = '{} tools_diar/diarization/make_rttm.py --rttm-channel 0 {} {}/{}.labels {}/{}.rttm' .format(python,segmentsfile,output_file,f,output_file,f)        
        os.system(cmd)
    

    def compute_score(self,rttm_gndfile,rttm_newfile,outpath,overlap):
      fold_local='services/'
      scorecode='score.py -r '
     
      # print('--------------------------------------------------')
      if not overlap:

          cmd=opt.which_python +' '+ fold_local + 'dscore-master/' + scorecode + rttm_gndfile + ' --ignore_overlaps --collar 0.25 -s ' + rttm_newfile + ' > ' + outpath + '.txt'
          os.system(cmd)
      else:
          cmd=opt.which_python + ' '+ fold_local + 'dscore-master/' + scorecode + rttm_gndfile + ' -s ' + rttm_newfile + ' > ' + outpath + '.txt'
          os.system(cmd)
      # print('----------------------------------------------------')
      # subprocess.check_call(cmd,stderr=subprocess.STDOUT)
      # print('scoring ',rttm_gndfile)
      bashCommand="cat {}.txt | grep OVERALL |awk '{{print $4}}'".format(outpath)
      output=subprocess.check_output(bashCommand,shell=True)
      return float(output.decode('utf-8').rstrip())
      # output = subprocess.check_output(['bash','-c', bashCommand])

    def compute_loss(self,A,minibatch,lamda):
        loss = 0.0
        weight = 1

        for m in minibatch:
            loss += -weight*A[m[0],m[1]]+lamda*(A[m[0],m[2]]+A[m[1],m[2]])+ 1.0
        # print('sum loss : ',loss)
        return loss/len(minibatch)

    def compute_minibatches(self,period,A,cluster,labels,mergeind=[],cleanind = []):
        triplets = []
        hard = 0
        random_sample = 1
        multiple = 0
        for ind,k in enumerate(cluster):
            neg = np.where(labels!=ind)[0]
            for i,a in enumerate(k[:-1]):                    
                for p in k[i+1:]:
                    Aavg = (A[a,neg]+A[p,neg])/2.0
                   
                    if hard:                            
                        neg_ind = np.argmax(Aavg)
                        fetch_negatives = neg[neg_ind]
                        triplets.append([a,p,fetch_negatives])
                    if random_sample:
                        max_10 = random.randint(0,len(Aavg)-1)
                        max_neg = min(max_10,len(Aavg)-1)
                        fetch_negatives = neg[max_neg]
                        triplets.append([a,p,fetch_negatives]) 
                    if multiple:
                        max_neg = np.random.randint(1, len(Aavg), size=(10,))
                        #neg_indices = np.argsort(Aavg,axis=None)[::-1][max_neg-1]
                        fetch_negatives = neg[max_neg]
                        for n in fetch_negatives:
                            triplets.append([a,p,n])

        
        random.shuffle(triplets)
        if len(triplets)==0:
            ValueError("No triplets generated!")
        triplets = np.array(triplets)
        N = len(triplets)
        N1=0
        
        need_batchsize = 0
  
        if need_batchsize:
            batchsize = opt.batchSize
            while not N1:
                N1 = N - (N % batchsize)
                batchsize = int(batchsize/2)
            batchsize = batchsize*2
            print('batchsize:',batchsize)
            minibatches = triplets[:N1].reshape(-1,batchsize,3)
        else:
            num_batches = min(opt.N_batches,N)
            N1 = N -(N % num_batches)
            batchsize = int(N1/num_batches)
            print('batchsize:',batchsize)
            minibatches = triplets[:N1].reshape(-1,batchsize,3)
        
        return minibatches,batchsize

    def compute_cluster(self,labels):
        unifull = np.unique(labels)
        ind = []
        for i,val in enumerate(unifull):
            ind.append((np.where(labels==val)[0]).tolist())
        return ind

    def dataloader_from_list(self):
        reco2utt = self.reco2utt
        D = self.xvecdimension

        channel = 1

        reco2utt=reco2utt.rstrip()
        f=reco2utt.split()[0]

        utts = reco2utt.split()[1:]
        
        if os.path.isfile(opt.xvecpath+f+'.npy'):
            system = np.load(opt.xvecpath+f+'.npy')

        else:
            
            utts = np.asarray(utts)
            system = np.empty((len(utts),D))
            for j,utt in enumerate(utts):
                xvec = np.load(opt.xvecpath+'/'+utt+'.npy')
                system[j] = xvec
            if not os.path.isdir(opt.xvecpath):
                os.makedirs(opt.xvecpath)
            np.save(opt.xvecpath+f+'.npy',system)

        x1_array=system[np.newaxis]
       

        data_tensor = torch.from_numpy(x1_array).float()
       

        return f,data_tensor

    

    def train_with_threshold(self,model_init):
        """
        train the network using triplet loss
        Threshold range : Decide intial number of clusters N0
        th = [0.0,0.1,0.2,0.3,0.4]
        ##############################
        Set following parameters here
        -----------------------------
        th_count : index of "th" array , selects the starting threshold (default: 1)
        stop_period: How many iterations to run (default: 1 i.e train for 1 iteration and then go till N*)
     
        ###############################

        saves the model,
        score matrix
        partial score matrix using previous labels

        period : iteration number from period 

        Parameters
        ----------
        weight initialization

        Returns
        -------
        None.

        """
       
        th_count = 1
        th = [0.0,0.1,0.2,0.3,0.4]
        set_dist = th[th_count] # setting threshold = 0.1
        stop_period  = min(1,th_count)


        model = self.model
        optimizer = self.optimizer
        loss_lamda_new =loss_lamda
        count = 0
        f,data = self.dataloader_from_list()
        print('---------------------------------------------------------')
        print('\nfilename:',f)
        
        inpdata =  data.float().to(device)
        nframe = data.size()[1] 
        
        # ground_labels=open('../Diarization_scores/output_labels_Harsha/'+dataset+'/threshold_0.75/labels_'+f).readlines()
        ground_labels=open('ALL_CALLHOME_GROUND_LABELS/'+dataset+'/threshold_0.75/labels_'+f).readlines()
        full_gndlist=[g.split()[1:] for g in ground_labels]
        gnd_list = np.array([g[0] for g in full_gndlist])
        uni_gnd_letter = np.unique(gnd_list)
        uni_gnd = np.arange(len(uni_gnd_letter))
        for ind,uni in enumerate(uni_gnd_letter):
            gnd_list[gnd_list==uni]=ind
        gnd_list = gnd_list.astype(int)

        clean_list = np.array([len(f) for f in full_gndlist])
        clean_ind =np.where(clean_list ==1)[0]
        overlap_ind = np.where(clean_list > 1)[0]
        print('file {} Overlap percentage: {}'.format(self.fname,100*len(overlap_ind)/nframe))
        n_prime = self.n_prime
        print('starting cluster: ',nframe)
       
       
        max_spks = n_prime
        period0len = n_prime
       
        stop_pt = nframe - max_spks # stop at 10 clusters
        period=0

        current_lr = opt.lr
 
        t=0
        labelfull_feed=np.arange(nframe)
        clusterlen_feed=[1]*len(labelfull_feed)

        while period < stop_period: # only 2 steps, increase for more iterations
           
            if period==0:
                model.eval()
               
                PCA_transform = model_init.compute_filewise_PCAtransform(self.pldamodel,inpdata) # original filewise PCA transform
                red_dimension = PCA_transform.shape[0]
                # to change dimension 
                # dimension = getattr(model, 'dimension')
                # setattr(model, 'filepca_out', nn.Linear(dimension,red_dimension,bias=False))
                model.init_weights(PCA_transform)

                output_model= model(inpdata)
                # output_model = model.compute_plda_affinity_matrix(inpdata_init)
                output_model1 = output_model.detach().cpu().numpy()[0]
                output_model = output_model1.copy()
                cosinefold = 'cosine_pca_baseline/{}_scores/cosine_scores/'.format(dataset)
                cosinefile = '{}/{}.npy'.format(cosinefold,f)
                
                if not os.path.isdir(cosinefold):
                    os.makedirs(cosinefold)
                
                if not os.path.isfile(cosinefile):
                    np.save(cosinefile,output_model)                

                clusterlen_old = clusterlen_feed.copy()
                labelfull_old = labelfull_feed.copy()
                n_clusters = max(period0len,n_prime)             

                # generate clusters using threshold set_threshold and n_clusters whichever reaches first
                myahc =ahc.clustering(n_clusters,clusterlen_feed, lamda,labelfull_feed,dist=set_dist)
                labelfull,clusterlen,_ = myahc.my_clustering_full(output_model)
                cluster = self.compute_cluster(labelfull)
                n_clusters = len(clusterlen)
                period0len = n_clusters
                labelfull_xvec = labelfull.copy()
                clusterlen_xvec = clusterlen.copy()
               
                check_clusterpurity(f, cluster, dataset,full_gndlist,labelfull)
               
                t = t+ nframe-n_clusters
            else:
               
                model.eval()
                output = model(inpdata)
               
                output = output.cpu().detach().numpy()[0]
                output_new = output.copy()
                nframe1 = period0len
                # use new model for period = 1 else use previous labels and then proceed 
                if period !=1: 
                    clusterlen_old = clusterlen.copy()
                    labelfull_old = labelfull.copy()
                    unifull = np.unique(labelfull_old)

                    clusterlist=[]
                    for val in unifull:
                        ind=np.where(labelfull_old==val)[0]
                        clusterlist.append(ind[0])
                        avg=np.sum(output_new[ind],axis=0)
                        output_new[ind[0]]=avg
                        output_new[:,ind[0]]=avg
                    output_new = output_new[np.ix_(clusterlist,clusterlist)]
                    nframe1 = output_new.shape[-1]
                n_clusters_old = n_clusters
                n_clusters = max(n_prime,max_spks)
                th_count = th_count-1
                set_dist = th[th_count]
                # use threshold in second training
                myahc =ahc.clustering(n_clusters, clusterlen_old,self.lamda,labelfull_old,dist=set_dist)
                labelfull,clusterlen,mergeind = myahc.my_clustering_full(output_new)
                
                cluster = self.compute_cluster(labelfull)
                loss_lamda_new  = loss_lamda_new + 0.1
                print('merging to ... ',min(nframe1-len(clusterlen),nframe1-n_prime))
                check_clusterpurity(f, cluster, dataset,full_gndlist,labelfull)
                n_clusters = len(clusterlen)
                model.init_weights(PCA_transform)
               
                output = model(inpdata)
               
                output = output.detach().cpu().numpy()[0]
                if n_clusters==max_spks:
                    t=stop_pt
                else:
                    t = nframe - n_clusters

            
            if period == 0:
              
                minibatches,batchsize = self.compute_minibatches(period,output_model1, cluster, labelfull)

               
                model.eval()
                output = model(inpdata)
               
                output = output.cpu().detach().numpy()[0]
                unifull = np.unique(labelfull)
                output_new = output.copy()
                clusterlist=[]
                for val in unifull:
                    ind=np.where(labelfull==val)[0]
                    clusterlist.append(ind[0])
                    avg=np.sum(output_new[ind],axis=0)
                    output_new[ind[0]]=avg
                    output_new[:,ind[0]]=avg
                output_new = output_new[np.ix_(clusterlist,clusterlist)]

                print('PCA intialisation with n_clusters:',n_clusters)
                valcluster,val_label = self.validate(output, count,n_clusters,clusterlen_feed,labelfull_feed,1)
                check_clusterpurity(f,valcluster,dataset,full_gndlist,val_label)
               
                valclusterlen = []
                for c in valcluster:
                    valclusterlen.append(len(c))
                print('clusterlen: ',valclusterlen)
                count +=1
                per_loss = opt.eta
                avg_loss = self.compute_loss(output, minibatches.reshape(-1,3), loss_lamda)
                print("\n[epoch %d] avg_loss: %.3f" % (0,avg_loss))
               
            else:
               
                minibatches,batchsize = self.compute_minibatches(period,output, cluster, labelfull,mergeind = mergeind)

                
                print('\n-------------------------------------------------------')

                print('Baseline Cosine DER with n_clusters:',n_clusters)
                valcluster,val_label=self.validate(output_model, count,n_clusters,clusterlen_feed,labelfull_feed,2)
                check_clusterpurity(f,valcluster,dataset,full_gndlist,val_label)
                count +=1
                print('Before training DER with n_clusters:',n_clusters)
                _,_=self.validate(output_new, count,n_clusters,clusterlen_old,labelfull_old,0)  # starting from previous merge
                count +=1
        
           
            for epoch in range(opt.epochs):

                model.train()

                model.zero_grad()
                self.optimizer.zero_grad()
                out_train = model(inpdata)
               
                triplet_avg_loss = self.compute_loss(out_train[0],minibatches.reshape(-1,3),loss_lamda_new)
                
                tot_avg_loss = triplet_avg_loss
                print("\n[epoch %d]  triplet_avg_loss: %.5f " % (epoch+1,triplet_avg_loss))
                if epoch == 0:
                    avg_loss = tot_avg_loss
                if tot_avg_loss < per_loss*avg_loss:
                    break
                tot_avg_loss.backward()
                self.optimizer.step()
               

            print('At t=',t,' clusters now:',len(cluster))

            print('System DER with n_clusters:',n_clusters)
            model.eval()
            output = model(inpdata)
            output = output.cpu().detach().numpy()[0]
            # np.save('output_random_negative_{}_{}.npy'.format(f,period),output)
            unifull = np.unique(labelfull_old)
            output_new = output.copy()
            clusterlist=[]
            for val in unifull:
                ind=np.where(labelfull_old==val)[0]
                clusterlist.append(ind[0])
                avg=np.sum(output_new[ind],axis=0)
                output_new[ind[0]]=avg
                output_new[:,ind[0]]=avg
            output_new = output_new[np.ix_(clusterlist,clusterlist)]
            valcluster,val_label=self.validate(output_new, count,n_clusters,clusterlen_old,labelfull_old,1) # evaluating on previous merge
            check_clusterpurity(f,valcluster,dataset,full_gndlist,val_label)
            valclusterlen = []
            for c in valcluster:
                valclusterlen.append(len(c))
            print('clusterlen: ',valclusterlen)
            count +=1
            period = period + 1


        print('Baseline Cosine DER with N*:',n_prime)
        _,_=self.validate(output_model1, count,n_prime,clusterlen_feed,labelfull_feed,2)
        count +=1
        self.final = 1
        model.eval()
        output1 = model(inpdata)
        # output1 = model.compute_plda_affinity_matrix(model(inpdata))
        output1 = output1.cpu().detach().numpy()
        output = output1[0]
        print('System DER with N*:',n_prime)
    
        valcluster,val_label=self.validate(output, count,n_prime,clusterlen_feed,labelfull_feed,1)
        check_clusterpurity(f,valcluster,dataset,full_gndlist,val_label)
        count +=1
        if n_clusters > n_prime:
            print('System DER using previous clusters with n_clusters:',n_clusters)
            self.forcing_label = 1
            if max_spks == period0len:
                labelfull_prev = labelfull_xvec
                clusterlen_prev = clusterlen_xvec
            else:
                labelfull_prev = labelfull_old
                clusterlen_prev = clusterlen_old
        
            unifull = np.unique(labelfull_prev)
            output_new = output.copy()
            clusterlist=[]
            for val in unifull:
                ind=np.where(labelfull_prev==val)[0]
                clusterlist.append(ind[0])
                avg=np.sum(output_new[ind],axis=0)
                output_new[ind[0]]=avg
                output_new[:,ind[0]]=avg
            output_new = output_new[np.ix_(clusterlist,clusterlist)]
        
            valcluster,val_label=self.validate(output_new, count,n_prime,clusterlen_prev,labelfull_prev,1)
            check_clusterpurity(f,valcluster,dataset,full_gndlist,val_label)
            count +=1
            
        print('Saving learnt parameters')
        matrixfold = "%s/cosine_scores/" % (opt.outf)
        savedict = {}
        savedict['output'] = output
        if n_clusters > n_prime:
            savedict['reduced_output'] = output_new
            savedict['labelfull'] = labelfull_prev
            savedict['clusterlen'] = clusterlen_prev
        if not os.path.isdir(matrixfold):
            os.makedirs(matrixfold)
        matrixfile = matrixfold + '/'+f+'.pkl'
        with open(matrixfile,'wb') as sf:
             pickle.dump(savedict,sf)
                

        print('\n-------------Saving model------------------------------------------')
        if not os.path.isdir(opt.outf+'/models/'):
            os.makedirs(opt.outf+'/models/')
        torch.save(model.state_dict(),opt.outf+'/models/'+f+'.pth')

    def validate(self,output_new, period,n_clusters,clusterlen,labelfull,flag):
            # lamda = 0
            f = self.fname
            overlap =0
            # bp()
            clusterlen_org = clusterlen.copy()
            if opt.threshold == 'None' or self.final==0:
                myahc =ahc.clustering(n_clusters, clusterlen_org,self.lamda,labelfull,dist=None)
            else:
                myahc =ahc.clustering(None, clusterlen_org,self.lamda,labelfull,dist=float(opt.threshold))
            labelfull,clusterlen,_ = myahc.my_clustering_full(output_new)
            print('clusterlen:',clusterlen)
            self.results_dict[f]=labelfull
            if self.final:
                if self.forcing_label:
                    out_file=opt.outf+'/'+'final_rttms_forced_labels/'
                else:
                    out_file=opt.outf+'/'+'final_rttms/'
            else:
                out_file=opt.outf+'/'+'rttms/'
            if not os.path.isdir(out_file):
                os.makedirs(out_file)
            outpath=out_file +'/'+f
            rttm_newfile=out_file+'/'+f+'.rttm'
            rttm_gndfile = 'rttm_ground/'+f+'.rttm'
            self.write_results_dict(out_file)
            # bp()
            der=self.compute_score(rttm_gndfile,rttm_newfile,outpath,0)
            if overlap:
                overlap_der = self.compute_score(rttm_gndfile,rttm_newfile,outpath,overlap)
            
            print("\n%s [period %d] DER: %.2f" % (self.fname,period, der))
           
            cluster = self.compute_cluster(labelfull)

            return cluster,labelfull


# hyper-parameters

keep_prob = 1 # 0.7
seed2=999
random.seed(seed2)
def main():

    xvecD=128
    pair_list = open(opt.reco2utt_list).readlines()
    filelen =len(pair_list)
    reco2num = open(opt.reco2num_spk).readlines()
    
    
    kaldimodel = pickle.load(open(opt.kaldimodel,'rb')) # PCA Transform and mean of heldout set
    ind = list(np.arange(filelen))
    random.shuffle(ind)

    
    seed=555
   
    pca_dim = 10
    for i in range(filelen):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print('pca_dim:',pca_dim)
        net_init = weight_initialization(kaldimodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
        model_init = net_init.to(device)
        net = Deep_Ahc_model(kaldimodel,dimension=xvecD,red_dimension=pca_dim,device=device)
        model = net.to(device)

        # Optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=opt.lr)
        

        # training

        reco2utt = pair_list[i]
        n_prime = int(reco2num[i].split()[1])
        fname = reco2num[i].split()[0]         
      
        print('output_folder:',opt.outf)
        ahc_obj = Deep_AHC(kaldimodel,fname,reco2utt,xvecD,model,optimizer,n_prime)
        ahc_obj.train_with_threshold(model_init)
        print('output_folder:',opt.outf)


if __name__ == "__main__":
    main()

