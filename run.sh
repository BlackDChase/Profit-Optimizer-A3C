#!/bin/bash
# Author  : BlackDChase
# Version : 0.2.0
cd modules
n=5
e=10
t=5
a=7
alr=0.01
clr=0.01
d=True
echo "'Model will start training with $n agents, training $e episodes of $t length, with $a actions, and Debugging set to $d, while actor learning rate and critic learining rate being at $alr and $clr respectivly, as Hyperparameters'"
python main.py -n=$n -e=$e -t=$t -a=$a -alr=$alr -clr=$clr -d=$d
#python main.py -a=7 -d=t -p="../Saved_model/Sun-04-04-21/"
echo "'Model Trained, extracting usefull info'"
sh extractLog.sh
echo "'Models saved with $n agents training $e episodes of $t length with $a actions as Hyperparameter'"
