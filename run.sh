#!/bin/bash
# Author  : BlackDChase
# Version : 0.1.3
cd modules
#python main.py -n=3 -e=100 -t=50 -a=7 -d=t -alr=0.01 -clr=0.01
python main.py -n=3 -e=10 -t=5 -a=3 -d=t -alr=0.01 -clr=0.01
#python main.py -a=7 -d=t -p="../Saved_model/Sun-04-04-21/"
sh extractLog.sh
echo "'Models saved with 3 agents training 10 episodes of 5 length with 5 actions'"
#echo "'Models saved with 3 agents training 100 episodes of 50 length with 15 actions'"
