"""
Main function
python main.py -n=3 -e=100 -t=50 -a=7 -d=t -alr=0.001 -clr=0.001 -p="../Saved_model/Sun-04-04-21/"
Parameters:
    - n = Number of agents
    - e = Number of episodes
    - t = Length of trajectory
    - a = Number of deviations in action
        - This means if a=5, 5*2+1 number of action [-12.5,-10 ... 0 ... +10,+12.5] percent change in price
    - d = If debug to be part of logs
    - alr = Actor Learning rate
    - clr = Critic Learning rate
    - p = Path of folder which contains PolicModel.py, CriticModel.pt
    - h = Help
#"""

__author__ = 'BlackDChase,MR-TLL'
__version__ = '0.1.1'
# Imports
# from TempEnv import TempEnv as ENV
from env import LSTMEnv as ENV
from Agent import GOD
import log
import sys

if __name__=="__main__":
    keywords={
        "n":2,
        "e":100,
        "t":5,
        "a":3,
        "d":False,
        "alr":1e-3,
        "clr":1e-3,
    }
    stateSize = 9
    log.info(f"stateSize = {stateSize}")
    
    


    for arg in sys.argv[1:]:
        key,value = arg.split("=")
        if key[1:]=="d":
            keywords[key[1:]] = True if "t" in value.lower() else False
            # For debug
        elif key[1:]=="p":
            keywords[key[1:]] = value
            # For path (saved model)
        else:
            keywords[key[1:]] = int(value)
        log.info(f"Parameter {keywords[key[1:]]} is {value}")
    if "h" in keywords:
        print("""
Main function
python main.py -n=3 -e=100 -t=50 -a=7 -d=t -alr=0.001 -clr=0.001 -p="../Saved_model/Sun-04-04-21/"
Parameters:
    - n     Number of agents
    - e     Number of episodes
    - t     Length of trajectory
    - a     Number of deviations in action
            This means if a=5, 5*2+1 number of action [-12.5,-10 ... 0 ... +10,+12.5] percent change in price
    - d     If debug to be part of logs
    - alr   Actor Learning rate
    - clr   Critic Learning rate
    - p     Path of folder which contains PolicModel.py, CriticModel.pt
    - h     Help""")

    if "p" not in keywords.keys():
        god = GOD(
            stateSize=stateSize,
            actionSpaceDeviation=keywords["a"],
            debug=keywords["d"],
            maxEpisode=keywords["e"],
            nAgent=keywords["n"],
            trajectoryLength=keywords["t"],
            alr=keywords["alr"],
            clr=keywords["clr"]
        )
        log.info("GOD inititated")
        actionSpace = god.getActionSpace()
        log.info(f"Action space: {actionSpace}")
      # env = ENV(stateSize,actionSpace)
        env=ENV.LSTM("ENV_MODEL/lstm_model.pt","../Dataset/13_Coloumn.csv")
        log.info("Environment inititated")
        god.giveEnvironment(env)
        log.info("Environment parsed, Boss inititated")
        threadCount=0
        try:
            god.train()
            god.saveModel("../Saved_model/")
        except Exception as catch:
            #log.debug(f"Terminaion Trace back {catch.with_traceback()}")
            threadCount+=1
            log.info(f"Terminated on a {threadCount}\t{catch}")
            log.info(f"Traceback for the {threadCount} Exception\t{sys.exc_info()}")
            print(f"{threadCount} thread Terminated, check log {log.name}")
    else:
        god = GOD(
            debug=keywords["d"],
            stateSize=stateSize,
            actionSpaceDeviation=keywords["a"],
            path=keywords["p"],
        )
        log.info("GOD inititated")
        actionSpace = god.getActionSpace()
        log.info(f"Action space: {actionSpace}")
        # env = ENV(stateSize,actionSpace)
        env=ENV.LSTM("ENV_MODEL/lstm_model.pt","../Dataset/13_Coloumn.csv")
        log.info("Environment inititated")
        god.giveEnvironment(env)
        log.info("Environment parsed, Boss inititated")
        """
        Live model use, not implimented
        #"""
