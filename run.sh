#!/bin/bash
# Author  : BlackDChase
# Version : 1.0.0

source bin/activate

getShut(){
    case $1 in
        [Yy]* ) echo "Will shut down once done."
            shut=1;;
        [Nn]* ) shut=2;;
        * ) shut=0;
            echo "Please answer yes or no.";;
    esac
    return $shut
}

cd modules


# Shutdown conditon
shutCon=$1
getShut $shutCon
shut=$?
while [[ "$shut" = 0 ]]; do
    read -p "Shutdown when done (Y/n): " shutCon
    getShut $shutCon
    shut=$?
done
s=0

 : '
# This is for Actuall training
n=50
e=500
t=55
a=8
s=1000
alr=0.002
clr=0.009
f=True
#d=True
# '

# : ' This is for test Training
n=3
e=10
t=10
a=8
s=100
alr=0.01
clr=0.07
f=True
d=True
# '

# : ' Training
echo "\"Model will start training with $n agents, training $e episodes of $t length, with $a actions, and Debugging set to $d, while actor learning rate and critic learining rate being at $alr and $clr respectivly, as Hyperparameters\""
folder="../Saved_model/" 
folder="$folder$(ls $folder -Art | grep "Olog" | tail -n 1)/"
p=$(echo "$folder$(ls $folder | grep "CritcModel.pt")" | rev | cut -b 14- | rev)
echo "Path is : $p"
python main.py -n=$n -e=$e -t=$t -a=$a -alr=$alr -clr=$clr -d=$d -p=$p -f=$f || echo "\"Stopped In Between\""

#For after training
echo "\"Model Trained and saved, extracting usefull info\""
./extractTrainLog.sh
echo "\"Extraction successfull, with $n agents training $e episodes of $t length with $a actions as Hyperparameter\""
# '

 : '
# This is for loading Latest trained model and testing it

#d=True
folder="../Saved_model/" 
folder="$folder$(ls $folder -Art | grep "Olog" | tail -n 1)/"
p=$(echo "$folder$(ls $folder | grep "CritcModel.pt")" | rev | cut -b 14- | rev)
echo "Path is : $p"
f=False
python main.py -n=$n -t=$t -a=$a -p=$p -s=$s -d=$d -f=$f -alr=$alr -clr=$clr  
echo "\"Model Tested for $fileName for $s time steps\""

# For After testing
fileName=$(echo $folder | cut -b 16- | rev | cut -b 2- | rev)
./extractTestLog.sh
echo "\"Extraction successfull, for $s timesteps\""
# '


# For shutting down system
if [[ "$shut" = 1 ]];then
    shutdown
fi
