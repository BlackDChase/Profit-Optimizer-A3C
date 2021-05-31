#!/bin/bash

# Author  : 'BlackDChase'
# Version : '1.5.3'

source bin/activate

# ./run.sh "y" "F" "F" "3" "2" "200"
shutCon=$1
f=$2
d=$3
m=$4
i=1
i=$5
intrupt=$6

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

getFine(){
    case $1 in
        [Tt]* ) echo "Finetuning true"
            shut=1;;
        [Ff]* ) echo "Finetuning false"
            shut=2;;
        * ) shut=0;
            echo "Please answer True or false";;
    esac
    echo $shut
    return $shut
}

getDebug(){
    case $1 in
        [Tt]* ) echo "Debugging true"
            shut=1;;
        [Ff]* ) echo "Debugging false"
            shut=2;;
        * ) shut=0;
            echo "Please answer True or false";;
    esac
    return $shut
}

getTestingMethodolgy(){
    case $1 in
        [nN0]*  ) echo "No testing at all"
            shut=0;;
        [oO1]* ) echo "Offline"
            shut=1;;
        [sS2]* ) echo "Sliding Window"
            shut=2;;
        [eE3]* ) echo "Episodic"
            shut=3;;
        [aA4]* ) echo "All three"
            shut=4;;
        * ) shut=5;
            echo "Please answer among 1,2,3,4";;
    esac
    return $shut
}

getItter(){
    if [[ $1 =~ ^[+-]?[0-9]+$ ]]; then
        return 1;
    fi
    return 0
}


# Checking if intrupt has recived a valid response
intruptValid(){
    case $1 in
        [0-9]* ) return 1;;
        [nN]* ) return 2;;
        * ) echo "Please answer among with seconds to wait for intrupt or No.";;
    esac
    return 0
}



# Shutdown conditon
getShut $shutCon
shut=$?
while [[ "$shut" = 0 ]]; do
    read -p "Shutdown when done (Y/n): " shutCon
    getShut "$shutCon"
    shut=$?
done

# Finetuning Conditon
getFine $f
fineCond=$?
while [[ "$fineCond" = 0 ]]; do
    read -p "Finetuning (T/f): " f
    getFine "$f"
    fineCond=$?
done

# Debugging Conditon
getDebug $d
debugCond=$?
while [[ "$debugCond" = 0 ]]; do
    read -p "Debugging (T/f): " d
    getDebug "$d"
    debugCond=$?
done


# Testing Conditon
getTestingMethodolgy $m
testCond=$?
while [[ "$testCond" = 5 ]]; do
    echo "Choose Testing Methodology:"
    echo "0. No test at all" 
    echo "1. Offline" # o/O works aswell
    echo "2. Sliding Window" # s/S 
    echo "3. Episodic" # e/E
    echo "4. All" # a/A
    read -p "Testing methodology [0,1,2,3,4]: " m
    getTestingMethodolgy "$m"
    testCond=$?
done

# Itterations of training
getItter $i
validity=$?
while [[ "$validity" = 0 ]];do
    read -p "Number of training Itterations : " i;
    getItter "$i"
    validity=$?
done



# Intrupt conditon
intruptValid $intrupt
validity=$?
while [[ "$validity" = 0 ]];do
    echo "Do you want to intrupt:"
    read -p "Should Intrupt time(s) /No  : " intrupt
    intruptValid "$intrupt"
    validity=$?
done


# Model Train and Test

# Finetuning, Debugging
echo "Will train for $i times while Finetuning is $f"
# Startin Test
while [[ $i != 0 ]];do
    echo "On run $i"
    ./train.sh "$f" "$d"
    i=$((i-1))
done

# Startin Test
# Fine tuning has to be disabled for testing.
# Else It will not load a previosuly saved model from $p and Start from Scratch
case $testCond in
    [0]*) echo "Training concludes";;
    [1]*) ./test.sh "o" "$d" "n";;
    
    [2]*) ./test.sh "s" "False" "$d"  "$intrupt";;
    [3]*) ./test.sh "e" "False" "$d"  "$intrupt";;
    [4]*) ./test.sh "o" "False" "$n"  "n";
          ./test.sh "s" "False" "$d"  "$intrupt"
          ./test.sh "e" "False" "$d"  "$intrupt";;
esac

# For shutting down system
if [[ "$shut" = 1 ]];then
    shutdown
fi
