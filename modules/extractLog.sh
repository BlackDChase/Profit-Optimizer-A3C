#!/bin/bash
name=$(date +%a-%d-%m-%y)
mkdir ../Saved_model/$name
cd "../Saved_model/$name"
mv "../logs/$name.tsv" "./"
awk "/\trewards/ {print}" $name.tsv > "rewardLog.tsv"
awk "/policy loss/ {print}" $name.tsv > "policyLossLog.tsv"
awk "/critic loss/ {print}" $name.tsv > "criticLossLog.tsv"
awk "/Advantage/ {print}" $name.tsv > "advantageLog.tsv"

mv "../*.pt" "./"
vim rewardLog.tsv  -c ':%s/.*rewards = //g' -c ':%s/\[//g'  -c ':%s/\]//g' -c ':wq'
vim policyLossLog.tsv  -c ':%s/.*loss = //g' -c ':wq'
vim criticLossLog.tsv  -c ':%s/.*loss = //g' -c ':wq'
vim advantageLog.tsv  -c ':%s/.*Advantage = //g' -c ':%s/\[//g'  -c ':%s/\]//g' -c ':wq'
echo "'Post Processing Logs was a success'"
python ../modules/postProcessing.py $name
echo "'Ploting was success'"
