#!/bin/bash
# Author  : BlackDChase
# Version : 0.4.0

name=$(ls ../logs -Art | grep "O.tsv" | tail -n 1)
IFS='.'
read -ra name <<< "$name"
name="${name[0]}"
mkdir ../Saved_test/$name
cd "../Saved_test/$name"
mv "../../logs/$name.tsv" "./"
awk "/\tA3C State/ {print}" $name.tsv > "A3CState.tsv"
awk "/\tNormal State/ {print}" $name.tsv > "NormalState.tsv"
awk "/\tA3C Profit/ {print}" $name.tsv > "A3CProfit.tsv"
awk "/\tNormal Profit/ {print}" $name.tsv > "NormalProfit.tsv"
awk "/\tDiff/ {print}" $name.tsv >"ProfitDiff.tsv"
echo "'Sub logs made'"

vim A3CProfit.tsv  -c ':%s/.*Profit = //g' -c ':wq'
vim A3CState.tsv  -c ':%s/.*\[//g'  -c ':%s/\].*//g' -c ':wq'
vim NormalProfit.tsv  -c ':%s/.*Profit = //g' -c ':wq'
vim NormalState.tsv  -c ':%s/.*\[//g'  -c ':%s/\].*//g' -c ':wq'
vim ProfitDiff.tsv -c ':%s/.*Diff = //g' -c ':wq'
echo "'Post Processing Logs was a success'"
python ../../modules/postTesting.py
#echo "'Ploting was success'"
