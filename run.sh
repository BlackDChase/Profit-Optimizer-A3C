#!/bin/bash
# Author  : BlackDChase
# Version : 0.4.0
cd modules

 : '
# This is for Actuall training
n=6
e=10000
t=25
a=7
s=1000
alr=0.003
clr=0.003
#d=True
# '

# : ' This is for test Training
n=4
e=100
t=25
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

 : '
# This is for loading trained model and testing it
# $f has to be same as the prefice of trained model
# $addr has to be same the saved_model folder of trained mode "2021-mm-dd-HH-MM -O" format
n=6
e=10000
t=25
a=7
s=1000
alr=0.003
clr=0.003
#d=True
f=$n"_"$e"_"$t"_"$a"_"$alr"_"$clr"_"
addr="2021-05-04-01-48-Olog"
p="../Saved_model/$addr/$f"

python main.py -a=$a -p=$p -s=$s -d=$d 
# For After testing
./extractTestLog.sh
echo "\"Model Tested for $addr/$f for $s time steps\""
# '
# For shutting doen system
#shutdown
