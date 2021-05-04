#!/bin/bash
# Author  : BlackDChase
# Version : 0.4.2
cd modules

# : '
# This is for Actuall training
n=6
e=5000
t=25
alr=0.003
clr=0.003
a=7
s=1000
#d=True
# '

 : ' This is for test Training
n=4
e=10
t=2
a=5
s=100
alr=0.06
clr=0.06
#d=True
# '

# : ' Training
echo "\"Model will start training with $n agents, training $e episodes of $t length, with $a actions, and Debugging set to $d, while actor learning rate and critic learining rate being at $alr and $clr respectivly, as Hyperparameters\""

python main.py -n=$n -e=$e -t=$t -a=$a -alr=$alr -clr=$clr -d=$d 

#For after training
echo "\"Model Trained, extracting usefull info\""
./extractTrainLog.sh
echo "\"Models saved with $n agents training $e episodes of $t length with $a actions as Hyperparameter\""
# '

# : '
# This is for loading Latest trained model and testing it

#d=True
folder="../Saved_model/" 
folder="$folder$(ls $folder -Art | grep 'Olog' | tail -n 1)/"
p=$(echo "$folder$(ls $folder | grep 'CritcModel.pt')" | rev | cut -b 14- | rev)
python main.py -a=$a -p=$p -s=$s -d=$d 
# For After testing
fileName=$(echo $folder | cut -b 16- | rev | cut -b 2- | rev)
./extractTestLog.sh
echo "\"Model Tested for $fileName for $s time steps\""
# '
# For shutting doen system
#shutdown
