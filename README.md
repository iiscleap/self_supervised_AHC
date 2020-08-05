# Deep Self-Supervised Hierarchical Clustering for Speaker Diarization

This repository is code used in our paper:
>  Deep Self-Supervised Hierarchical Clustering for Speaker Diarization  
> Prachi Singh, Sriram Ganapathy  
> Accepted in Interspeech 2020
## Overview
The recipe consists of:
  - extracting x-vectors using pretrained model
  - Performing self-supervised clustering for diarization on Callhome set
  - Evaluating results using Diarization error rate (DER)
 
## Prerequisites
The following packages are required to run the baseline.

- [Python](https://www.python.org/) >= 3.6
- [Kaldi](https://github.com/kaldi-asr/kaldi)
- Pytorch >= 0.4
- [dscore](https://github.com/nryant/dscore)

## Getting started

  - clone the repository:
  ```sh
git clone https://github.com/iiscleap/self_supervised_AHC.git
```
- Install [Kaldi](https://github.com/kaldi-asr/kaldi). 
If you are a Kaldi novice, please consult the following for additional documentation:
    - [Kaldi tutorial](http://kaldi-asr.org/doc/tutorial.html)
    - [Kaldi for Dummies tutorial](http://kaldi-asr.org/doc/kaldi_for_dummies.html)
- Go to cloned repository and copy kaldi path in ``path.sh`` given as:
 ```sh
 $ local_dir="Full_path_of_cloned_repository"
 $ echo "export KALDI_ROOT="/path_of_kaldi_directory/kaldi" >> $local_dir/path.sh
 $ echo "export KALDI_ROOT="/path_of_kaldi_directory/kaldi" >> $local_dir/tools_dir/path.sh
 ```
 - Input x-vectors features are obtained using [Kaldi Callhome-diarization](https://kaldi-asr.org/models/m6). Pre-trained x-vector model and plda model including global mean and PCA transform needed  for training are given in [``tools_diar/callhome_xvector_models``](https://github.com/iiscleap/self_supervised_AHC/tree/master/tools_diar/callhome_xvector_models):
-  Performance is evaluated using [dscore](https://github.com/nryant/dscore). Download all the required dependencies in the same python environment.
 ## Implementation 
 ##### X-vectors Extraction
 - This step is to run kaldi diarization pipeline till x-vector extraction using pre-trained model
 - Additionally it will convert x-vectors in ark format into numpy format to run in pytorch. It will also convert kaldi plda model into pickle format.
 - Replace "data_root" with path of callhome dataset in [``tools_diar/run_extract_xvectors.sh``](https://github.com/iiscleap/self_supervised_AHC/blob/master/tools_diar/run_extract_xvectors.sh)
 - Run following commands:
 ```sh
 $ local_dir="Full_path_of_cloned_repository"
 $ cd $local_dir/tools_diar
 $ bash run_extract_xvectors.sh
 ```
 ##### DNN Training
 - xvec_ahc_train.py is code for DNN training
 - run_xvec_ahc.sh calls DNN training script
 - Update training parameters in run_xvec_ahc.sh
 **NOTE** that by default Kaldi scripts are configured for execution on a grid using a submission engine such as SGE or Slurm. If you are running the recipes on a single machine, make sure to edit ``cmd.sh`` and ``tools_dir/cmd.sh`` so that the line
```sh
   $ export train_cmd="queue.pl"
```
reads
```sh
   $ export train_cmd="run.pl"
```  
 - Execute following commands:
 ```sh
 $ local_dir="Full_path_of_cloned_repository"
 $ cd $local_dir
 $ bash run_xvec_ahc.sh $local_dir  --TYPE parallel --nj <number of jobs> --which_python <python_with_all_installed_libraries>
 ```
 **Note**: --TYPE parallel (when running multiple jobs simultaneoulsy)
##### Evaluation
- Diarization Error Rate is used as performance metric 
- Scripts in [dscore](https://github.com/nryant/dscore) generates filewise DER. 
- Go to cloned repo and run following command for evaluation
```sh
$ local_dir="Full_path_of_cloned_repository"
$ cd $local_dir/tool_diar
$ bash gen_rttm.sh --stage <1/2> --modelpath <path of model to evaluate>
```
**Note**: --stage 1 (using ground truth number of speakers), --stage 2 (using threshold based number of clusters)

##### Output
- Generates ``der.scp`` in modelpath which contains filewise DER and other metric like JER.
 

