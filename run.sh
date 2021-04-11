#!/bin/bash
cd modules
python main.py -n=3 -e=100 -t=50 -a=7 -d=t -alr=0.001 -clr=0.001
# python main.py -a=7 -d=t -p="../Saved_model/Sun-04-04-21/"
sh extractLog.sh
echo "'Models saved with 3 agents training 100 episodes of 50 length with 15 actions'"
