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
__version__ = '0.1.3'
# Input from outside
import log
import sys
keywords={
        "n":1,
        "e":10,
        "t":5,
        "a":5,
        "d":True,
        "alr":1e-3,
        "clr":1e-3,
    }
stateSize = 13
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
        keywords[key[1:]] = float(value)
    log.info(f"Parameter {key[1:]} is {value}")
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
        
"""
Making the action space
"""
n=int(keywords['a']//2)
if keywords['a']%2==0:    
    actionSpace = [i for i in range(-n*2,n*2+1,2)]
    actionSpace.pop(len(actionSpace)//2)
else:
    actionSpace = [i/10 for i in range(-n*25,n*25+1,25)]
keywords['a']=actionSpace
   
    
# Imports
"""
from TempEnv import TempEnv as ENV
"""
from lstm import LSTM
from env import LSTMEnv as ENV
#"""
from Agent import GOD

if __name__=="__main__":
    if "p" not in keywords.keys():
        god = GOD(
            stateSize=int(stateSize),
            actionSpace=keywords["a"],
            debug=keywords["d"],
            maxEpisode=int(keywords["e"]),
            nAgent=int(keywords["n"]),
            trajectoryLength=int(keywords["t"]),
            alr=keywords["alr"],
            clr=keywords["clr"]
        )
        log.info("GOD inititated")
        log.info(f"Action space: {actionSpace}")

        #"""
        output_size = 13
        input_dim = output_size
        hidden_dim = 128
        layer_dim = 1
        model = LSTM(output_size, input_dim, hidden_dim, layer_dim)
        model.loadM("ENV_MODEL/lstm_model.pt")
        log.info(f"LSTM Model = {model}")
        #env=ENV(model,"../Dataset/13_columns.csv")
        env=ENV(model=model,dataset_path="../Dataset/normalized_13_columns.csv",actionSpace=actionSpace)
        env.reset()
        """
        env = ENV(stateSize,actionSpace)
        #"""
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
            stateSize=int(stateSize),
            actionSpace=keywords["a"],
            path=keywords["p"],
        )
        log.info("GOD inititated")
        actionSpace = god.getActionSpace()
        log.info(f"Action space: {actionSpace}")
        #"""
        model=LSTM("ENV_MODEL/lstm_model.pt")
         #env=ENV(model,"../Dataset/13_columns.csv")
        env=ENV(model=model,dataset_path="../Dataset/normalized_13_columns.csv",actionSpace=actionSpace)
        """
        env = ENV(stateSize,actionSpace)
        #"""
        log.info("Environment inititated")
        god.giveEnvironment(env)
        log.info("Environment parsed, Boss inititated")
        """
        Live model use, not implimented
        #"""
