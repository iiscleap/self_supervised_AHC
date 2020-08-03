#!/bin/bash
# @author: prachi singh 
# @email: prachisingh@iisc.ac.in 
main_dir=$1
dataset=callhome1  # interchange for training callhome2 files
dataset2=callhome2
outf=tf_events/xvec_ahc_trained/${dataset}_scores/ # output folder path
TYPE=parallel # training parallely multiple utterances
nj=40 # number of jobs for parallelizing
mkdir -p $outf/

. ./cmd.sh
. ./tools_diar/path.sh

. ./tools_diar/utils/parse_options.sh

main_dir=$1

if [ -z "$1" ]; then
	echo "need main_directory full path as argument"
	echo " Set arguments for training in the code"
	echo "Usage : bash run_xvec_ahc.sh --TYPE <parallel/None> --nj <number of jobs> full path of main directory "
	exit 1
fi

JOB=1
if [ $TYPE == "parallel" ]; then 
    if [ ! -d lists/$dataset/tmp/split$nj ] || [ ! "$(ls -A lists/$dataset/tmp/split$nj/1)" ]; then
        cd tools_diar
    	utils/split_data_mine.sh $main_dir/lists/$dataset/tmp $nj || exit 1;
        cd ../
    fi
	#$train_cmd JOB=1:$nj $outf/log/Deep_AHC.JOB.log \
	python xvec_ahc_train.py \
	--gpuid '0' \
	--batchSize 64 \
	--N_batches 1 \
	--epochs 10 \
	--lr 1e-3 \
    --eta 0.5 \
	--lamda 0.0 \
	--gamma 0.4 \
	--dataset $dataset \
	--outf $outf \
	--xvecpath ../Diarization_scores/swbd_diar/xvectors_npy/${dataset}/ \
	--filetrain_list lists/$dataset/tmp/split$nj/$JOB/dataset.list \
	--reco2utt_list lists/$dataset/tmp/split$nj/$JOB/spk2utt \
	--threshold 'None' \
	--segments lists/$dataset/segments_xvec \
	--reco2num_spk lists/$dataset/tmp/split$nj/$JOB/reco2num_spk \
	--kaldimodel lists/$dataset2/plda_${dataset2}.pkl

else
	python xvec_ahc_train.py \
	--gpuid '0' \
	--batchSize 64 \
	--N_batches 1 \
	--epoch 10 \
	--lr 1e-3 \
    --eta 0.5 \
	--lamda 0.0 \
	--gamma 0.4 \
	--dataset $dataset \
	--outf $outf \
	--xvecpath tools_diar/xvectors_npy/${dataset}/ \
	--filetrain_list lists/$dataset/${dataset}.list \
	--reco2utt_list lists/$dataset/tmp/spk2utt \
	--threshold 'None' \
	--segments lists/$dataset/segments_xvec \
	--reco2num_spk swbd_diar/data/$dataset/reco2num_spk \
	--kaldimodel lists/$dataset2/plda_${dataset2}.pkl

fi
