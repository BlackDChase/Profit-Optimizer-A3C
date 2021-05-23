#!/bin/bash

# Author  : 'BlackDChase'
# Version : '1.4.0'

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
        [nN]*  ) echo "No testing at all"
            shut=5;;
        [oO0]* ) echo "Offline"
            shut=1;;
        [sS1]* ) echo "Sliding Window"
            shut=2;;
        [eE2]* ) echo "Episodic"
            shut=3;;
        [aA3]* ) echo "All three"
            shut=4;;
        * ) shut=0;
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
    getShut $shutCon
    shut=$?
done

# Finetuning Conditon
getFine $f
fineCond=$?
while [[ "$fineCond" = 0 ]]; do
    read -p "Finetuning (T/f): " f
    getFine $f
    fineCond=$?
done

# Debugging Conditon
getDebug $d
debugCond=$?
while [[ "$debugCond" = 0 ]]; do
    read -p "Debugging (T/f): " d
    getDebug $d
    debugCond=$?
done


# Testing Conditon
getTestingMethodolgy $m
testCond=$?
while [[ "$testCond" = 0 ]]; do
    echo "Choose Testing Methodology:"
    echo "n. No test at all" 
    echo "0. Offline" # o/O works aswell
    echo "1. Sliding Window" # s/S 
    echo "2. Episodic" # e/E
    echo "3. All" # a/A
    read -p "Testing methodology [0,1,2,3]: " m
    getTestingMethodolgy $m
    testCond=$?
done

# Itterations of training
getItter $i
validity=$?
while [[ "$validity" = 0 ]];do
    read -p "Number of training Itterations : " i;
    getItter $i
    validity=$?
done



# Intrupt conditon
intruptValid $intrupt
validity=$?
while [[ "$validity" = 0 ]];do
    echo "Do you want to intrupt:"
    read -p "Should Intrupt time(s) /No  : " intrupt
    intruptValid $intrupt
    validity=$?
done


# Model Train and Test

# Finetuning, Debugging
echo "Will train for $i times while Finetuning is $f"
while [[ $i != 0 ]];do
    ./train.sh "$f" "$d"
    i=$((i-1))
done

case $m in
    [0]*) ./test.sh "o" "$d";;
    
    [1]*) intruptValid $intrupt
        validity=$?
        if [[ "$validity" = "1" ]]; then
            ./test.sh "s" "$d" &
            childPid=($!)
            sleep $intrupt
            kill -SIGINT $childPid
        else
            ./test.sh "s" "$d"
             echo "Will not Intrupt, have to do it manually"
        fi
        echo "Intrupt success";;
    [2]*) intruptValid $intrupt
        validity=$?
        if [[ "$validity" = "1" ]]; then
            ./test.sh "o" "$d" &
            childPid=($!)
            sleep $intrupt
            kill -SIGINT $childPid
        else
            ./test.sh "s" "$d"
             echo "Will not Intrupt, have to do it manually"
        fi
        echo "Intrupt success";;
    [3]*) ./test.sh "o" ;
        intruptValid $intrupt
        validity=$?
        if [[ "$validity" = "1" ]]; then
            ./test.sh "s" "$d" &
            childPid=($!)
            sleep $intrupt
            kill -SIGINT $childPid
            ./test.sh "o" "$d" &
            childPid=($!)
            sleep $intrupt
            kill -SIGINT $childPid
        else
            ./test.sh "s" "$d"
             echo "Will not Intrupt, have to do it manually"
        fi
        echo "Intrupt success";;
    [4]*) echo "Training concludes"
esac

# For shutting down system
if [[ "$shut" = 1 ]];then
    shutdown
fi
