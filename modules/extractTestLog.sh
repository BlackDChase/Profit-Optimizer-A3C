#!/bin/bash
# Author  : 'BlackDChase'
# Version : '1.4.9'
s=$?
name=$(ls ../logs -Art | grep "Olog.tsv" | tail -n 1)
IFS='.'
read -ra name <<< "$name"
name="${name[0]}"
mkdir ../Saved_test/$name
cd "../Saved_test/$name"
mv "../../logs/$name.tsv" "./"

echo "$name" > README.md
awk "/\tTesting Mode\t/{print}" $name.tsv >> "README.md"
awk "/\tPara/ {print}" $name.tsv >> "README.md"
vim README.md -c ':%s/.*\tTesting Mode/# Testing Mode/g' -c ':wq'
vim README.md -c ':%s/.*Para/- Para/g' -c ':wq'

    awk "/\tA3C State/ {print}" $name.tsv > "A3CState.tsv"
    awk "/\tA3C Profit/ {print}" $name.tsv > "A3CProfit.tsv"
    awk "/\tNormal State/ {print}" $name.tsv > "NormalState.tsv"
    awk "/\tNormal Profit/ {print}" $name.tsv > "NormalProfit.tsv"
    awk "/\tDiff/ {print}" $name.tsv >"ProfitDiff.tsv"
    echo "Sub logs made"
    vim NormalProfit.tsv  -c ':%s/.*Profit = //g' -c ':wq'
    vim NormalState.tsv  -c ':%s/.*\[//g'  -c ':%s/\].*//g' -c ':wq'
    vim ProfitDiff.tsv -c ':%s/.*Diff = //g' -c ':wq'

    vim A3CProfit.tsv  -c ':%s/.*Profit = //g' -c ':wq'
    vim A3CState.tsv  -c ':%s/.*\[//g'  -c ':%s/\].*//g' -c ':wq'

    awk "/\trewards/ {print}" $name.tsv > "rewardLog.tsv"
    awk "/Policy loss/ {print}" $name.tsv > "policyLossLog.tsv"
    awk "/Critic loss/ {print}" $name.tsv > "criticLossLog.tsv"
    awk "/\tAdvantage/ {print}" $name.tsv > "advantageLog.tsv"
    awk "/\tState set/ {print}" $name.tsv > "stateLog.tsv"

    echo "Loss logs are made"
    vim policyLossLog.tsv  -c ':%s/.*=//g' -c ':%s/.*\[//g'  -c ':%s/\].*//g' -c ':wq'
    vim rewardLog.tsv  -c ':%s/.*=//g' -c ':%s/.*\[//g'  -c ':%s/\].*//g' -c ':wq'
    vim criticLossLog.tsv  -c ':%s/.*=//g' -c ':%s/.*\[//g'  -c ':%s/\].*//g' -c ':wq'
    vim advantageLog.tsv  -c ':%s/.*=//g' -c ':%s/.*\[//g'  -c ':%s/\].*//g' -c ':wq'
    vim stateLog.tsv  -c ':%s/.*=//g' -c ':%s/.*\[//g'  -c ':%s/\].*//g' -c ':wq'


echo "Post Processing Logs was a success"

# New models moved to $name
critic=$(ls ../ -Art | grep "CritcModel." | tail -n 1)
actor=$(ls ../ -Art | grep "PolicyModel." | tail -n 1)
mv "../$critic" "./"
mv "../$actor" "./"
echo "New models moved to $name"

python ../../modules/postTesting.py 
echo "Ploting was success"
