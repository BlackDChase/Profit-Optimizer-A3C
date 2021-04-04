"""
Main function
python ar.py -n=3 -e=100 -t=50 -a=7 -d=t
Parameters:
    - n = Number of agents
    - e = Number of episodes
    - t = Length of trajectory
    - a = Number of deciations in action
        - This means if a=5, 5*2+1 number of action [-12.5,-10 ... 0 ... +10,+12.5] percent change in price
    - d = If debug to be part of logs
#"""

__author__ = 'BlackDChase,MR-TLL'
__version__ = '0.1.1'
# Imports
from TempEnv import TempEnv as ENV
from Agent import GOD
import log
import sys

if __name__=="__main__":
    keywords={
        "n":2,
        "e":100,
        "t":25,
        "a":5,
        "d":False
    }
    stateSize = 9
    log.info(f"stateSize = {stateSize}")

    for arg in sys.argv[1:]:
        key,value = arg.split("=")
        if key[1:]=="d":
            keywords[key[1:]] = True if "t" in value.lower() else False
        else:
            keywords[key[1:]] = int(value)
        log.info(f"Parameter {keywords[key[1:]]} is {value}")

    god = GOD(
        maxEpisode=keywords["e"],
        nAgent=keywords["n"],
        debug=keywords["d"],
        trajectoryLength=keywords["t"],
        stateSize=stateSize,
        actionSpaceDeviation=keywords["a"],
    )
    log.info("GOD inititated")
    actionSpace = god.getActionSpace()
    log.info(f"Action space: {actionSpace}")
    env = ENV(stateSize,actionSpace)
    log.info("Environment inititated")
    god.giveEnvironment(env)
    log.info("Environment parsed, Boss inititated")
    threadCount=0
    try:
        god.train()
    except Exception as catch:
        #log.debug(f"Terminaion Trace back {catch.with_traceback()}")
        threadCount+=1
        log.info(f"Terminated on a {threadCount}\t{catch}")
        log.info(f"Traceback for the {threadCount} Exception\t{sys.exc_info()}")
        print("{threadCount} thread Terminated, check log ",log.name)
