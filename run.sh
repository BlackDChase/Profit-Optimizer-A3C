#!/bin/bash
cd modules
python main.py -n=3 -e=100 -t=50 -a=7 -d=t
sh extractLog.sh
echo "'Models saved with 3 agents training 100 episodes of 50 length with 15 actions'"
