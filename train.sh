#!/bin/bash

# Author  : 'BlackDChase'
# Version : '1.4.7'

# ./train.sh "True" "False"

# Finetuning and debugging
f=$1
d=$2

cd modules

# : ' This is default
n=30
e=200
t=75
a=9
alr=0.01
clr=0.07
g=0.6
# '

# : '
# This is for Actuall training
#n=50
e=1000
#t=200
a=8
#alr=0.002
#clr=0.009
#g=0.9
# '


# Path, if using previously trained model
folder="../Saved_model/" 
folder="$folder$(ls $folder -Art | grep "Olog" | tail -n 1)/"    
if [[ "$f" = True ]]; then
    if [[ -f "$folder$(ls $folder | grep "CritcModel.pt")" ]]; then
        p=$(echo "$folder$(ls $folder | grep "CritcModel.pt")" | rev | cut -b 14- | rev)
        echo "Loading Model from : $p"
    fi
else
    f=False
    echo "Training from Scratch"
fi


# Training
echo "Model will start training with $n agents, training $e episodes of $t length, with $a actions, and Debugging set to $d, while actor learning rate and critic learining rate being at $alr and $clr respectivly, as Hyperparameters"

python main.py -n=$n -e=$e -t=$t -a=$a -alr=$alr -clr=$clr -d=$d -p=$p -f=$f -g=$g 


#For after training
echo "Model Trained and saved, extracting usefull info"
./extractTrainLog.sh

echo "Extraction successfull, with $n agents training $e episodes of $t length with $a actions as Hyperparameter"
