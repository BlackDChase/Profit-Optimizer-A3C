#/bin/bash

# Shut condition
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

isValidOption(){
    case $1 in
        [1]* ) return 1;;
        [2]* ) return 2;;
        * ) echo "Please answer among 1 or 2.";;
    esac
    return 0
}

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
echo "Choose wait option:"
echo "1. Process to die"
echo "2. Seconds"
option=$2
isValidOption $option
validity=$?
while [[ "$validity" = 0 ]];do
    read -p "Option [1,2]  : " option
        isValidOption $option
        validity=$?
done
case $option in
    [1]*) read -p "Process ID  : " pid;
        while true;do
            x=$(ps -l $pid)
            if [[ "$x" =~ "$pid" ]];then
                sleep 60
            else
                break
            fi
        done;;
    [2]*) read -p "Sleep time  : " t
        sleep $t;;
esac

# Running RUN
echo "Running the algo"
./run.sh > Out.log "$shutCon"
echo "Output saved in Out.log"
