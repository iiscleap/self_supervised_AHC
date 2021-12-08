#Usage : this script generates filewise rttm 
#bash compute_rttm.sh <dataset name> <path where rttm is generated> <rttm name>
mkdir -p $2/final_rttms
cat ../lists/$1/${1}.list | while read i; do
grep $i $2/$3 > $2/final_rttms/${i}.rttm
done

