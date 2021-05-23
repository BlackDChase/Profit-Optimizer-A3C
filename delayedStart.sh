#!/bin/bash
# Author  : 'BlackDChase'
# Version : '1.3.9'

# How to run --------------------------------------------------------------------------------------

# To Run            Shutdown afterwards     After end of process    Process ID      Intrupt time
# ./delayedStart.sh "y"                     "1"                     "3990"          "3000"

# To Run            Keepworking             After some time         Wait Time       Dont intrupt
# ./delayedStart.sh "N"                     "2"                     "1000"          "n"

# Example run:
#./delayedStart.sh "y" "1"  "19450"  "2000" >> OUTDelayed.log


# Functions ---------------------------------------------------------------------------------------

# Shut condition
getShut(){
    case $1 in
        [Yy]* ) echo "Will shut down once done."
            return 1;;
        [Nn]* ) echo "Will not shutdown"
            return 2;;
        * ) echo "Please answer yes or no.";;
    esac
    return 0
}

# Checking if waiting has recived a valid response
waitValid(){
    case $1 in
        [1]* ) echo "Will wait for process to die"
            return 1;;
        [2]* ) echo "Will wait for some time"
            return 2;;
        * ) echo "Please answer among 1 or 2.";;
    esac
    return 0
}

waitCommandValidity(){
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

run(){
    # Running RUN
    echo "Running the algo"
    ./run.sh > OUT.log "$1"
}



# Condition checks --------------------------------------------------------------------------------

# Shutdown conditon
shutCon=$1
getShut $shutCon
shut=$?
while [[ "$shut" = 0 ]]; do
    read -p "Shutdown when done (Y/n): " shutCon
    getShut $shutCon
    shut=$?
done

# Waiting conditon
waitOption=$2
waitValid $waitOption
validity=$?
while [[ "$validity" = 0 ]];do
    echo "Choose wait waitOption:"
    echo "1. Process to die"
    echo "2. Seconds"
    read -p "waitOption [1,2]  : " waitOption
    waitValid $waitOption
    validity=$?
done

# Waiting PID/sleeping time
waitCommand=$3
waitCommandValidity $waitCommand
validity=$?
while [[ "$validity" = 0 ]];do
    case $waitOption in
        [1]* ) read -p "Enter Waiting for death of Pid : " waitCommand;;
        [2]* ) read -p "Enter Wait Time  : " waitCommand;;
    esac
    waitValid $waitCommand
    validity=$?
done

# Intrupt conditon
intrupt=$4
intruptValid $intrupt
validity=$?
while [[ "$validity" = 0 ]];do
    echo "Do you want to intrupt:"
    read -p "Should Intrupt time(s) /No  : " intrupt
    intruptValid $intrupt
    validity=$?
done

# Waiting
case $waitOption in
    [1]*) while true;do
            x=$(ps -l $waitCommand)
            if [[ "$x" =~ "$waitCommand" ]];then
                sleep 60
            else
                break
            fi
        done;;
    [2]*) sleep $waitCommand;;
esac

# Actual Run --------------------------------------------------------------------------------------

# Running training on parallel thread
run $shutCon &
childPid=($!)

# Sending KeyboardIntrupt to the run
intruptValid $intrupt
validity=$?
case $validity in
    [1]* ) echo "Will wait for $4 before Intrupt"
        sleep $4
        kill -SIGINT $childPid;;
    [2]*) echo "Will not intrupt $childPid"
        echo "To manually intrupt 'kill -SIGINT $childPid'";;
    *) echo "Invalid Option";;
esac

echo "Output will be saved in OUT.log"
