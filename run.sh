#!/bin/bash
# Author  : BlackDChase
# Version : 1.0.0
cd modules

while true;do
    read -p "Shutdown when done (Y/n): " shut
    case $shut in
        [Yy]* ) shut=true;break;;
        [Nn]* ) shut=false;break;;
        * ) echo "Please answer yes or no.";;
    esac
done
# : '
# This is for Actuall training
n=20
e=500
t=75
a=8
s=250
alr=0.0016
clr=0.0075
f=True
#d=True
# '

# : ' This is for test Training
n=3
e=10
t=10
a=8
s=100
alr=0.01
clr=0.07
f=True
d=True
# '

# : ' Training
echo "\"Model will start training with $n agents, training $e episodes of $t length, with $a actions, and Debugging set to $d, while actor learning rate and critic learining rate being at $alr and $clr respectivly, as Hyperparameters\""
folder="../Saved_model/" 
folder="$folder$(ls $folder -Art | grep 'Olog' | tail -n 1)/"
#p=$(echo "$folder$(ls $folder | grep 'CritcModel.pt')" | rev | cut -b 14- | rev)
echo "Path is : $p"
python main.py -n=$n -e=$e -t=$t -a=$a -alr=$alr -clr=$clr -d=$d -p=$p -f=$f || echo "\"Stopped In Between\""

#For after training
echo "\"Model Trained and saved, extracting usefull info\""
./extractTrainLog.sh
echo "\"Extraction successfull, with $n agents training $e episodes of $t length with $a actions as Hyperparameter\""
# '

# : '
# This is for loading Latest trained model and testing it

#d=True
folder="../Saved_model/" 
folder="$folder$(ls $folder -Art | grep 'Olog' | tail -n 1)/"
p=$(echo "$folder$(ls $folder | grep 'CritcModel.pt')" | rev | cut -b 14- | rev)
python main.py -a=$a -p=$p -s=$s -d=$d 
echo "\"Model Tested for $fileName for $s time steps\""

# For After testing
fileName=$(echo $folder | cut -b 16- | rev | cut -b 2- | rev)
./extractTestLog.sh
echo "\"Extraction successfull, for $s timesteps\""
# '
# For shutting down system
if $shut;then
    shutdown
fi
