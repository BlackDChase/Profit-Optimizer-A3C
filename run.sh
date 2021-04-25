#!/bin/bash
# Author  : BlackDChase
# Version : 0.2.0
cd modules
n=6
e=500
t=50
a=5
alr=0.001
clr=0.001
#d=True
echo "'Model will start training with $n agents, training $e episodes of $t length, with $a actions, and Debugging set to $d, while actor learning rate and critic learining rate being at $alr and $clr respectivly, as Hyperparameters'"

# For actuall Run
python main.py -n=$n -e=$e -t=$t -a=$a -alr=$alr -clr=$clr -d=$d

# For test Run
#python main.py -n=3 -e=10 -t=5 -a=5 -alr=0.1 -clr=0.1 -d=True

# For loading model, for testing
#python main.py -a=7 -d=t -p="../Saved_model/Sun-04-04-21/"

echo "'Model Trained, extracting usefull info'"
./extractLog.sh
echo "'Models saved with $n agents training $e episodes of $t length with $a actions as Hyperparameter'"

# For shutting doen system
#shutdown
