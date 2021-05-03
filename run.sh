#!/bin/bash
# Author  : BlackDChase
# Version : 0.3.8
cd modules

# : '
# This is for Actuall training
n=6
e=400
t=30
a=5
s=1000
alr=0.001
clr=0.001
#d=True
# '

 : ' This is for test Training
n=3
e=25
t=5
a=5
s=100
alr=0.1
clr=0.1
d=True
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
e=400
t=30
a=5
s=1000
alr=0.001
clr=0.001
f=$n"_"$e"_"$t"_"$a"_"$alr"_"$clr"_"
addr="2021-05-03-18-29 -O"
p="../Saved_model/$addr/$f"
#d=True
python main.py -a=$a -d=$d -p=$p -s=$s
# For After testing
./extractTestLog.sh
echo "\"Model Tested for $addr/$f for $s time steps\""
# '
# For shutting doen system
#shutdown
