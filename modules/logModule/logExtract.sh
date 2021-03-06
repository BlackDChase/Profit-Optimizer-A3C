#!/bin/bash
name=$(date +%a-%d-%m-%y)
mkdir ../../Saved_model/$name
cd "../../Saved_model/$name"
mv "../log.tsv" "./"
awk "/\trewards/ {print}" log.tsv > "rewardLog.tsv"
awk "/loss/ {print}" log.tsv > "lossLog.tsv"
mv "../*.pt" "./"
vim rewardLog.tsv  -c ':%s/.*rewards = //g' -c ':%s/\[//g'  -c ':%s/\]//g' -c ':wq'
vim lossLog.tsv  -c ':%s/.*loss = //g' -c ':wq'
echo "'Post Processing Logs was a success'"
python ../../modules/logModule/postProcessing.py $name
echo "'Ploting was success'"
