#!/bin/bash


# This script is just a copy of the run.sh in the egs/callhome_diarization/v2/,
# which was used to train the system in this resouce.  Since you the x-vector
# DNN is already trained, you will most likely just need to run the stages for
# computing features and extracting and clustering x-vectors.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
data_root=/export/corpora/NIST/LDC2001S97/   # callhome dataset path
stage=0
nnet_dir=callhome_xvector_models/exp/xvector_nnet_1a/  # path of xvector model

# Prepare datasets
if [ $stage -le 0 ]; then

  # Prepare the Callhome portion of NIST SRE 2000.
  local/make_callhome.sh $data_root data/

fi

# Prepare features
if [ $stage -le 1 ]; then
  # The script local/make_callhome.sh splits callhome into two parts, called
  # callhome1 and callhome2.  Each partition is treated like a held-out
  # dataset, and used to estimate various quantities needed to perform
  # diarization on the other part (and vice versa).
  for name in callhome1 callhome2; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/$name exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/$name
  done
  
  # if required kaldi VAD in case oracle SAD is not used
  # for name in callhome1 callhome2; do
  #   sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  #     data/$name exp/make_vad $vaddir
  #   utils/fix_data_dir.sh data/$name
  # done

  # This writes features to disk after applying the sliding window CMN.
  # Although this is somewhat wasteful in terms of disk space, for diarization
  # it ends up being preferable to performing the CMN in memory.  If the CMN
  # were performed in memory (e.g., we used --apply-cmn true in
  # diarization/nnet3/xvector/extract_xvectors.sh) it would need to be
  # performed after the subsegmentation, which leads to poorer results.
  for name in callhome1 callhome2; do
    local/nnet3/xvector/prepare_feats.sh --nj 40 --cmd "$train_cmd" \
      data/$name data/${name}_cmn exp/${name}_cmn
      utils/fix_data_dir.sh data/${name}_cmn
  done

 

  # Extract x-vectors
if [ $stage -le 2 ]; then
  # Extract x-vectors for the two partitions of callhome.
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj 40 --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $nnet_dir \
    data/callhome1_cmn exp/xvectors_callhome1

  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj 40 --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $nnet_dir \
    data/callhome2_cmn exp/xvectors_callhome2

fi
# convert x-vectors from ark to numpy , convert kaldi plda, transform.mat, mean.vec into pickle format
#and copy spk2utt,utt2spk, segments in lists folder

if [ $stage -le 3 ]; then
# converts x-vectors from ark to numpy and convert kaldi models into pickle
for dataset in callhome1 callhome2; do
    srcdir=exp/xvectors_$dataset   # path of xvectors.scp
    python ../services/read_scp_write_npy_embeddings.py vec $srcdir/xvectors.scp xvectors_npy/${dataset}/
    python ../services/convert_kaldi_to_pkl.py $nnet_dir/xvectors_$dataset $dataset
    python ../services/generate_groundtruth_label_sequence.py --dataset $dataset --segmentsfile ../lists/$dataset/tmp/segments \
    --labelsfiledir ../ALL_CALLHOME_GROUND_LABELS/$dataset/threshold_0.75/ --ground_truth_rttm data/$dataset/rttm
done

# copy spk2utt,utt2spk, segments in lists folder required for training
for dataset in callhome1 callhome2; do
    srcdir=exp/xvectors_$dataset   # path of xvectors.scp

    mkdir -p ../lists/$dataset/tmp
    cp $srcdir/spk2utt ../lists/$dataset/tmp/spk2utt
    cp $srcdir/utt2spk ../lists/$dataset/tmp/utt2spk
    cp $srcdir/segments ../lists/$dataset/tmp/segments
    cp data/$dataset/reco2num_spk ../lists/$dataset/reco2num_spk
    cp data/$dataset/reco2num_spk ../lists/$dataset/tmp/reco2num_spk

    awk '{print $1}' $srcdir/spk2utt > ../lists/$dataset/${dataset}.list
    cp ../lists/$dataset/$dataset.list ../lists/$dataset/tmp/dataset.list


    # store segments filewise in folder segments_xvec
    mkdir -p ../lists/segments_xvec
    cat ../lists/$dataset/${dataset}.list | while read i; do
        grep $i ../lists/$dataset/tmp/segments > ../lists/segments_xvec/${i}.segments
    done
done

fi