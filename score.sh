#Usage: This script generate filewise DER , ignoring overlaps and collar : 0.25s
# bash score.sh <path where rttm is stored> <file containing path of ground truth filewise rttm> 

path=$1
find $path/final_rttms/*.rttm > $path/sys.scp
python services/dscore-master/score.py --ignore_overlaps --collar 0.25 -R $2 -S $path/sys.scp > $path/der.scp 2> metrics_dev.stderr
grep OVERALL $path/der.scp

