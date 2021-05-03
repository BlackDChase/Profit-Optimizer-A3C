#!/bin/bash
# Author  : BlackDChase
# Version : 0.4.0

name=$(ls ../logs -Art | grep "O.tsv" | tail -n 1)
IFS='.'
read -ra name <<< "$name"
name="${name[0]}"
mkdir ../Saved_model/$name
cd "../Saved_model/$name"
mv "../../logs/$name.tsv" "./"
awk "/\trewards/ {print}" $name.tsv > "rewardLog.tsv"
awk "/Policy loss/ {print}" $name.tsv > "policyLossLog.tsv"
awk "/Critic loss/ {print}" $name.tsv > "criticLossLog.tsv"
awk "/\tAdvantage/ {print}" $name.tsv > "advantageLog.tsv"
awk "/\tState set/ {print}" $name.tsv > "stateLog.tsv"
echo "'Sub logs made'"

critic=$(ls ../ -Art | grep "CritcModel." | tail -n 1)
actor=$(ls ../ -Art | grep "PolicyModel." | tail -n 1)

mv "../$critic" "./"
mv "../$actor" "./"

vim policyLossLog.tsv  -c ':%s/.*loss = //g' -c ':wq'
vim rewardLog.tsv  -c ':%s/.*tensor//g' -c ':%s/(\[//g'  -c ':%s/\].*//g' -c ':wq'
vim criticLossLog.tsv  -c ':%s/.*loss = //g' -c ':wq'
vim advantageLog.tsv  -c ':%s/.*tensor//g' -c ':%s/(\[//g'  -c ':%s/\].*//g' -c ':wq'
vim stateLog.tsv -c ':%s/.*set = //g' -c ':wq'
echo "'Post Processing Logs was a success'"
python ../../modules/postTraining.py
#echo "'Ploting was success'"
