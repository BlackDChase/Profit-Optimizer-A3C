#!/bin/bash

# Author  : 'BlackDChase'
# Version : '1.4.2'

# Default Hyperparameters
testVarient=$1
f=$2
d=$3
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
    python main.py -t=$t -a=$a -p=$p -s=$s -d=$d -f=$f -alr=$alr -clr=$clr -g=$g ;
    return 1
}

slidingOnlineTest(){
    echo "Sliding Window testing"
    m=False
    t=$1
    g=$2
    alr=$3
    clr=$4
    python main.py -t=$t -a=$a -p=$p -s=$s -d=$d -f=$f -alr=$alr -clr=$clr -g=$g -m=$m ;
    return 1
}

episodicOnlineTest(){
    echo "Episodic testing"
    m=True
    t=$1
    g=$2
    alr=$3
    clr=$4
    python main.py -t=$t -a=$a -p=$p -s=$s -d=$d -f=$f -alr=$alr -clr=$clr -g=$g -m=$m ;
    return 1
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


case $testVarient in
    [oO0]* ) # Hyperparameters for Offline
        s=1000
        offlineTest $s;;
    [sS1]* ) # Hyperparameters for Sliding Window Online
        t=75
        alr=0.002
        clr=0.009
        g=0.8
        slidingOnlineTest $t $g $alr $clr;;
    [eE2]* ) # Hyperparameters for Episodic Online
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
        episodicOnlineTest $t $g $alr $clr;;
    * ) echo "Invalid varient";;
esac


testSuccess=$?
if [[ "$testSuccess" = 1 ]]; then
    fileName=$(echo $folder | cut -b 16- | rev | cut -b 2- | rev)
    echo "Model Tested for $fileName for $s time steps"
fi


# For After testing
echo "Sliding window is $m"
./extractTestLog.sh "$s"
echo "Extraction successfull, for $s timesteps"
