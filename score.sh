path=$1
find $path/final_rttms/*.rttm > $path/$3

python services/dscore-master/score.py --ignore_overlaps --collar 0.25 -R $2 -S $path/$3 > $path/$4
grep OVERALL $path/$4 

