   
. ./cmd.sh
. ./path.sh


stage=1
modelpath=../tf_events/xvec_ahc_folder
which_python=python # python envorionment with required libraries installed
. utils/parse_options.sh

# using oracle number of speakers 

if [ $stage -eq 1 ]; then
  lamda=0.0
  
  # modelpath=../tf_events/xvec_ahc_learnablePCA_cosine_ahc_multi_m29
 
  # Need initialization or not
  prev=0
  reco2num_spk=reco2num_spk
  
  labels=labels
  for dataset in callhome1 callhome2 ;do
    score_cosine_path=$modelpath/${dataset}_scores/

    ./my_cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
    --which_python $which_python \
    --reco2num_spk data/$dataset/$reco2num_spk \
   --lamda $lamda --score_path $score_cosine_path/cosine_scores/ \
    --usinginit $prev  --score_file ../lists/${dataset}/${dataset}.list \
          exp/${dataset}/ $score_cosine_path/cosine_${labels}_init${prev}_oracle/ 

    md-eval-22.pl -1 -c 0.25 -r \
       data/$dataset/ref.rttm -s $score_cosine_path/cosine_${labels}_init${prev}_oracle/rttm 2> $score_cosine_path/cosine_${labels}_init${prev}_oracle/DER.log \
       > $score_cosine_path/cosine_${labels}_init${prev}_oracle/DER.txt

       der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
      $score_cosine_path/cosine_${labels}_init${prev}_oracle/DER.txt)
      echo "der $dataset: $der"
      
  done

  mkdir -p $modelpath/results

  mkdir -p $modelpath/callhome/

  cat $modelpath/callhome1_scores/cosine_${labels}_init${prev}_oracle/rttm $modelpath/callhome2_scores/cosine_${labels}_init${prev}_oracle/rttm > $modelpath/callhome/${labels}_init${prev}_oracle_rttm

  cat $modelpath/callhome/${labels}_init${prev}_oracle_rttm | md-eval-22.pl -1 -c 0.25 -r \
     data/callhome/fullref.rttm -s - 2> $modelpath/results/${labels}_init${prev}_oracle_threshold.log \
     > $modelpath/results/${labels}_init${prev}_oracle_DER_threshold.txt


  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $modelpath/results/${labels}_init${prev}_oracle_DER_threshold.txt)
  
  echo "der callhome: $der"

  
  bash ../compute_rttm.sh callhome $modelpath/callhome/ ${labels}_init${prev}_oracle_rttm 
  bash ../score.sh $modelpath/callhome/ ../ref_callhome.scp

  echo "filewise der at $modelpath/callhome/der.scp"

fi


# thresholding my clustering

if [ $stage -eq 2 ]; then

   lamda=0.0
  
   # modelpath=../tf_events/xvec_ahc_learnablePCA_cosine_ahc_multi_m29
   
   prev=0

   for dataset in callhome1 callhome2 ;do
      #break
      score_cosine_path=$modelpath/${dataset}_scores/
      mkdir -p $score_cosine_path/tuning
      best_der=100
      best_threshold=0
      for threshold in -0.2 -0.1 0 0.1 0.2 ; do         
        
        ./my_cluster.sh --cmd "$train_cmd" --nj 20 \
        --which_python $which_python \
        --threshold $threshold \
        --lamda $lamda --score_path $score_cosine_path/cosine_scores/ \
        --usinginit $prev --score_file ../lists/${dataset}/${dataset}.list \
          exp/${dataset}/ $score_cosine_path/cosine_labels_init${prev}_t$threshold/ 

        md-eval-22.pl -1 -c 0.25 -r data/$dataset/ref.rttm \
         -s $score_cosine_path/cosine_labels_init${prev}_t$threshold/rttm \
         2> $score_cosine_path/tuning/init${prev}_t${threshold}.log \
         > $score_cosine_path/tuning/init${prev}_t${threshold}

        der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
          $score_cosine_path/tuning/init${prev}_t${threshold})
        echo "der: $der"

        if [ $(perl -e "print ($der < $best_der ? 1 : 0);") -eq 1 ]; then
          best_der=$der
          best_threshold=$threshold
        fi
        echo "best der $best_der at $best_threshold"
      done
      echo "$best_threshold" > $score_cosine_path/tuning/init${prev}_best_threshold
      echo "$dataset - best der : $best_der at best threshold : $best_threshold"
   done
   
   dataset=callhome1
   dataset2=callhome2
   score_cosine_path=$modelpath/${dataset}_scores/
   ./my_cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
   -- which_python $which_python \
    --threshold $(cat $modelpath/${dataset2}_scores/tuning/init${prev}_best_threshold) \
   --lamda $lamda --score_path $score_cosine_path/cosine_scores/ \
    --usinginit $prev  --score_file ../lists/${dataset}/${dataset}.list \
          exp/${dataset}/ $score_cosine_path/cosine_labels_init${prev}_best/ 


   dataset=callhome2
   dataset2=callhome1
    score_cosine_path=$modelpath/${dataset}_scores/
    ./my_cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
    -- which_python $which_python \
    --threshold $(cat $modelpath/${dataset2}_scores/tuning/init${prev}_best_threshold) \
    --lamda $lamda --score_path $score_cosine_path/cosine_scores/ \
    --usinginit $prev  --score_file ../lists/${dataset}/${dataset}.list \
          exp/${dataset}/ $score_cosine_path/cosine_labels_init${prev}_best/ 

   
   mkdir -p $modelpath/results

   mkdir -p $modelpath/callhome/

   cat $modelpath/callhome1_scores/cosine_labels_init${prev}_best/rttm $modelpath/callhome2_scores/cosine_labels_init${prev}_best/rttm > $modelpath/callhome/init${prev}_rttm

   cat $modelpath/callhome/init${prev}_rttm | md-eval-22.pl -1 -c 0.25 -r \
     data/callhome/fullref.rttm -s - 2> $modelpath/results/init${prev}_threshold.log \
     > $modelpath/results/init${prev}_DER_threshold.txt


   der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $modelpath/results/init${prev}_DER_threshold.txt)
   echo "der callhome: $der"

   bash ../compute_rttm.sh callhome $modelpath/callhome/ init${prev}_rttm
   bash ../score.sh $modelpath/callhome/ ../ref_callhome.scp

   echo "filewise der at $modelpath/callhome/der.scp"

fi


