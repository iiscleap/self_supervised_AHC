

# gen_rttm_fold=../generated_rttm/rttm_callhome1_newclustering/model3_sigmoid_masked_plda_org/
gen_rttm_fold=generated_rttm_new/callhome1/transformer1/
labels_fold=labels_transformer1_callhome1/
# labels_fold=/home/data1/SRE18/harshav/Xvec_Transformer1/saves/callhome_outputlabels/
mkdir -p $gen_rttm_fold
# cat lists/callhome1/callhome1.list | while read reco;
# do
reco=iaaa
    /home/prachis/anaconda3/envs/amytorch/bin/python services/make_rttm_for_overlap.py \
    --rttm-channel 0  \
    lists/callhome1/segments_1.5/${reco}.segments \
    output_labels/$labels_fold/labels_${reco} \
    ${gen_rttm_fold}/${reco}_v1.rttm

    sort -k 4n  ${gen_rttm_fold}/${reco}_v1.rttm >  ${gen_rttm_fold}/${reco}.rttm
    rm  ${gen_rttm_fold}/${reco}_v1.rttm
# done
# reco='iaqh'
# python make_rttm.py \
# --rttm-channel 0  \
# ../lists/callhome1/segments_1.5/${reco}.segments \
# ../output_labels/labels_ground_callhome1_all/${reco}.labels \
# ${gen_rttm_fold}/${reco}_v1.rttm

# sort -k 4n  ${gen_rttm_fold}/${reco}_v1.rttm >  ${gen_rttm_fold}/${reco}.rttm
# rm  ${gen_rttm_fold}/${reco}_v1.rttm