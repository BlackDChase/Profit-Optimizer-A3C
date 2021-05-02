#!/bin/bash
# Author  : BlackDChase
# Version : 0.3.5
cd modules
n=6
e=25
t=5
a=1
alr=0.001
clr=0.001
s=1000
#d=True
echo "'Model will start training with $n agents, training $e episodes of $t length, with $a actions, and Debugging set to $d, while actor learning rate and critic learining rate being at $alr and $clr respectivly, as Hyperparameters'"

# For testing
python main.py -n=$n -e=$e -t=$t -a=$a -alr=$alr -clr=$clr -d=$d

# For temp test Run
#python main.py -n=3 -e=100 -t=10 -a=3 -alr=0.1 -clr=0.1 -s=100 -d=True

# For loading model, for testing
#python main.py -a=$a -d=t -p="../Saved_model/Sun-04-04-21/" -s=$s

echo "'Model Trained, extracting usefull info'"
./extractLog.sh
echo "'Models saved with $n agents training $e episodes of $t length with $a actions as Hyperparameter'"

# For shutting doen system
#shutdown
