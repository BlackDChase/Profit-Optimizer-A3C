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
    - s = Times steps to test for
    - h = Help
#"""

__author__ = 'BlackDChase,MR-TLL'
__version__ = '0.4.0'
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
        "s":100,
        "p":None
    }
stateSize = 13
log.info(f"stateSize = {stateSize}")
arg,value,key='','',''
try:
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
except:
    print(key,arg,value)
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
    - s     Times steps to test for
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
    if keywords["p"] is None:
        print("Model Will be trained")
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
        print("Master Agent Made")
        log.info("GOD inititated")
        log.info(f"State Size = {int(stateSize)}")
        log.info(f"Action Space = {keywords['a']}")
        log.info(f"Debug = {keywords['d']}")
        log.info(f"Max Episode = {int(keywords['e'])}")
        log.info(f"Number of Agent = {int(keywords['n'])}")
        log.info(f"Trajectory Length = {int(keywords['t'])}")
        log.info(f"Actor Learning Rate = {keywords['alr']}")
        log.info(f"Critic Learning Rate = {keywords['clr']}")

        threadCount=0
        try:
            print("Model starting it's training")
            god.train()
            god.saveModel("../Saved_model")
        except KeyboardInterrupt:
            god.saveModel("../Saved_model")
        except Exception as catch:
            #log.debug(f"Terminaion Trace back {catch.with_traceback()}")
            threadCount+=1
            log.info(f"Terminated on a {threadCount}\t{catch}")
            log.info(f"Traceback for the {threadCount} Exception\t{sys.exc_info()}")
            print(f"{threadCount} thread Terminated, check log")
        print("Trained")
    else:
        print("Model will be tested")
        god = GOD(
            debug=keywords["d"],
            stateSize=int(stateSize),
            actionSpace=keywords["a"],
            path=keywords["p"],
        )
        print("Master Agent Made")
        log.info("GOD inititated")
        log.info(f"State Size = {int(stateSize)}")
        log.info(f"Action Space = {keywords['a']}")
        log.info(f"Debug = {keywords['d']}")
        log.info(f"Path of Model = {keywords['p']}")
        log.info(f"Steps tested for = {keywords['s']}")
        #"""

        output_size = 13
        input_dim = output_size
        hidden_dim = 128
        layer_dim = 1
        model = LSTM(output_size, input_dim, hidden_dim, layer_dim,debug=keywords["d"])
        model.loadM("ENV_MODEL/lstm_model.pt")

        env=ENV(
            model=model,
            dataset_path="../datasets/normalized_weird_13_columns_with_supply.csv",
            actionSpace=actionSpace,
            debug=keywords["d"],
        )
        print("Environment inititated")
        """
        env = ENV(stateSize,actionSpace)
        #"""
        log.info("Environment inititated")
        god.giveEnvironment(env)
        log.info("Environment parsed, Boss inititated")
        log.info(f"ENV LSTM: {model}")

        # Testing
        time=int(keywords['s'])
        a3cStates = god.test(time=time)
        normalStates = god.getNormalStates(time=time)
        a3cProfit,normalProfit,diff = god.compare(a3cState=a3cStates,normalState=normalStates)

        # Plotting
        for i in range(len(a3cStates)):
            log.info(f"A3C State = {a3cStates[i]}")
            log.info(f"Normal State = {normalStates[i]}")
        for i in range(len(a3cProfit)):
            log.info(f"A3C Profit = {a3cProfit[i]}")
            log.info(f"Normal Profit = {normalProfit[i]}")
            log.info(f"Diff = {diff[i]}")
        print("Tested")
