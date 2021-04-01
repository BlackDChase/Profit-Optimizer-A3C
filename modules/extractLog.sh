#!/bin/bash
name=$(date +%a-%d-%m-%y)
mkdir ../Saved_model/$name
cd "../Saved_model/$name"
mv "../logs/$name.tsv" "./"
awk "/\trewards/ {print}" $name.tsv > "rewardLog.tsv"
awk "/loss/ {print}" $name.tsv > "lossLog.tsv"
mv "../*.pt" "./"
vim rewardLog.tsv  -c ':%s/.*rewards = //g' -c ':%s/\[//g'  -c ':%s/\]//g' -c ':wq'
vim lossLog.tsv  -c ':%s/.*loss = //g' -c ':wq'
echo "'Post Processing Logs was a success'"
python ../modules/postProcessing.py $name
echo "'Ploting was success'"
