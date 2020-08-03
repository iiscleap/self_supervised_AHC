import os
import argparse


def read_arguments():
    parser = argparse.ArgumentParser(description="Deep_AHC")
    parser.add_argument("--gpuid", type=str, default=0, help="GPU id to run the code")
    parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
    parser.add_argument("--N_batches", type=int, default=100, help="Number of batches")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--eta", type=float, default=0.5, help="value of eta parameter") 
    parser.add_argument("--lamda", type=float, default=0.0, help="value of lambda parameter")
    parser.add_argument("--gamma", type=float, default=0.4, help="value of gamma parameter")
    parser.add_argument("--outf", type=str, default="cnn_models/", help='path of log files')
    parser.add_argument("--outf_trained", type=str, default="cnn_models/", help='path of pretrained model')
    parser.add_argument("--xvecpath", type=str, default='callhome1_scores_truncated/', help='systems_scores_path')
    parser.add_argument("--filetrain_list", type=str, default=None, help='path of trainfiles list')
    parser.add_argument("--reco2utt_list", type=str, default=None, help='spk2utt')
    parser.add_argument("--segments", type=str, default=None, help='segments')
    parser.add_argument("--threshold", type=str, default=None, help='number of cluster or threshold')
    parser.add_argument("--reco2num_spk", type=str, default=None, help='reco2num_spk')
    parser.add_argument("--baselineder", type=str, default=None, help='der_pickle')
    parser.add_argument("--kaldimodel", type=str, default=None, help='path of plda pickle model')
    parser.add_argument("--dataset", type=str, default='Callhome1', help='name of dataset')    
    opt = parser.parse_args()

    return opt
