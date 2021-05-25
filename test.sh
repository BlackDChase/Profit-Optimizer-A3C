#!/bin/bash

# Author  : 'BlackDChase'
# Version : '1.4.8'

# ./test.sh "e" "False" "False" "10"
# Default Hyperparameters
if [[ "$PWD" != $VIRTUAL_ENV ]];then
    source bin/activate
fi

testVarient=$1
f=$2
d=$3
intrupt=$4
a=8
s=0
t=75
alr=0.002
clr=0.009
g=0.6



cd modules

offlineTest(){
    s=$1
    echo "Offline testing"
    echo "$s"
    python main.py -t=$t -a=$a -p=$p -s=$s -d=$d -f=$f -alr=$alr -clr=$clr -g=$g 
}

slidingOnlineTest(){
    echo "Sliding Window testing"
    m=False
    t=$1
    g=$2
    alr=$3
    clr=$4
    python main.py -t=$t -a=$a -p=$p -s=$s -d=$d -f=$f -alr=$alr -clr=$clr -g=$g -m=$m 
}

episodicOnlineTest(){
    echo "Episodic testing"
    m=True
    t=$1
    g=$2
    alr=$3
    clr=$4
    python main.py -t=$t -a=$a -p=$p -s=$s -d=$d -f=$f -alr=$alr -clr=$clr -g=$g -m=$m
}


# Path, if using previously trained model
folder="../Saved_model/" 
folder="$folder$(ls "$folder" -Art | grep "Olog" | tail -n 1)/"
if [[ "$f" = False ]]; then
    p=$(echo "$folder$(ls "$folder" | grep "CritcModel.pt")" | rev | cut -b 14- | rev)
    echo "Loading Model from : $p"
else 
    echo "Training/Testing from Scratch"
fi

childPid=1

case $testVarient in
    [oO0]* ) # Hyperparameters for Offline
        s=1000
        offlineTest "$s" &
        methodPid=$!
        childPid=$(pgrep -P $methodPid)
        intrupt="n"
        ;;
    [sS1]* ) # Hyperparameters for Sliding Window Online
        t=75
        alr=0.002
        clr=0.009
        g=0.8
        slidingOnlineTest "${t}" "${g}" "${alr}" "${clr}" & 
        methodPid=$!
        childPid=$(pgrep -P $methodPid);;
    [eE2]* ) # Hyperparameters for Episodic Online
        echo "Episodic Case"
         : ' 
        t=2000
        g=0.99
        alr=0.05
        clr=0.1
         : '
        t=75
        g=0.8
        alr=0.01
        clr=0.05
        # '
        episodicOnlineTest "${t}" "${g}" "${alr}" "${clr}" &
        methodPid=$!
        childPid=$(pgrep -P $methodPid)
        ;;
    * ) echo "Invalid varient";;
esac

if [[ "$childPid" != 1 ]]; then
    case $intrupt in
        [0-9]*) echo "Waiting for $childPid for $intrupt"
                sleep $intrupt
                kill -SIGINT $childPid
                echo "$childPid Concluded";;
        [nN]*)  echo "Waiting for $childPid"
                echo "Will not Intrupt, have to do it manually"
                sleep 10
                wait $childPid;; 
    esac
    fileName=$(echo $folder | cut -b 16- | rev | cut -b 2- | rev)
    echo "Model Tested for $fileName for $s time steps"
fi


# For After testing
echo "Sliding window is $m"
./extractTestLog.sh 
echo "Extraction successfull, for $s timesteps"
