#!/bin/bash

# Author  : 'BlackDChase'
# Version : '1.2.7'

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
# : ' This is default
n=3
e=10
t=10
a=8
s=0
alr=0.01
clr=0.07
g=0.9
f=True
d=True
# '

# : '
# This is for Actuall training
n=50
e=1000
t=55
a=8
s=1000
alr=0.002
clr=0.009
g=0.99
f=True
d=False
# '


# : ' Training
echo "Model will start training with $n agents, training $e episodes of $t length, with $a actions, and Debugging set to $d, while actor learning rate and critic learining rate being at $alr and $clr respectivly, as Hyperparameters"
folder="../Saved_model/" 
folder="$folder$(ls $folder -Art | grep "Olog" | tail -n 1)/"
p=$(echo "$folder$(ls $folder | grep "CritcModel.pt")" | rev | cut -b 14- | rev)
echo "Path is : $p"
python main.py -n=$n -e=$e -t=$t -a=$a -alr=$alr -clr=$clr -d=$d -p=$p -f=$f || echo "Stopped In Between"

#For after training
echo "Model Trained and saved, extracting usefull info"
./extractTrainLog.sh
echo "Extraction successfull, with $n agents training $e episodes of $t length with $a actions as Hyperparameter"
# '


 : '
# This is for loading Latest trained model and testing it

#d=True
folder="../Saved_model/" 
folder="$folder$(ls $folder -Art | grep "Olog" | tail -n 1)/"

#p=$(echo "$folder$(ls $folder | grep "CritcModel.pt")" | rev | cut -b 14- | rev)
s=0
s=1000
m=True
m=False

echo "Path is : $p"
f=False
python main.py -n=$n -t=$t -a=$a -p=$p -s=$s -d=$d -f=$f -alr=$alr -clr=$clr  
echo "Model Tested for $fileName for $s time steps"

# For After testing
fileName=$(echo $folder | cut -b 16- | rev | cut -b 2- | rev)
./extractTestLog.sh "$s"
echo "Extraction successfull, for $s timesteps"
# '


# For shutting down system
if [[ "$shut" = 1 ]];then
    shutdown
fi


: ' 
    - n     Number of agents
    - e     Number of episodes
    - t     Length of trajectory
    - a     Number of deviations in action
            This means if a=5, 5*2+1 number of action [-12.5,-10 ... 0 ... +10,+12.5] percent change in price
    - d     If debug to be part of logs
    - alr   Actor Learning rate
    - clr   Critic Learning rate
    - p     Path of folder which contains PolicModel.py, CriticModel.pt
    - s     Times steps to test for, if 0, will test in  online mode until KeyboardInterrupt
    - f     Finetune trained Model
    - g     gamma
    - m     True: Episodic Method, False: Sliding Window Method
    - h     Help
'
