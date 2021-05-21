"""
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
    - s     Times steps to test for, if 0, will test in  online mode until KeyboardInterrupt.
            I is not initialised, will be parsed when called god.test().
    - f     Finetune trained Model
    - h     Help
#"""

__author__ = 'BlackDChase,MR-TLL'
__version__ = '1.2.3'

# Input from outside
import log
import sys
keywords={
        "n":1,
        "e":10,
        "t":5,
        "a":5,
        "d":"",
        "alr":1e-3,
        "clr":1e-3,
        "s":0,
        "p":None,
        "f":"",
    }

stateSize = 13
log.info(f"stateSize = {stateSize}")
arg,value,key='','',''

# Parsing arguments
try:
    for arg in sys.argv[1:]:
        key,value = arg.split("=")
        # For debug and fine tuning 
        if key[1:]=="d" or key[1:]=="f":
            keywords[key[1:]] = True if "t" in value.lower() else False
        # For assigning the path of the policy and critic models
        elif key[1:]=="p":
            keywords[key[1:]] = value
        # Rest arguments
        else:
            keywords[key[1:]] = float(value)
        log.info(f"Parameter {key[1:]} is {value}")
except:
    print(key,arg,value)

# To show more info about the list of parameters that can be passed during runtime when -h is enabled
if "h" in keywords:
    print("""
Main function
python main.py -n=3 -e=100 -t=50 -a=7 -d=t -alr=0.001 -clr=0.001 -f=True -p='../Saved_model/yyyy-mm-dd-HH-mm-Olog/'
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
    - s     Times steps to test for, if 0, will test in  online mode until KeyboardInterrupt
    - f     Finetune trained Model
    - h     Help
""")
    sys.exit()


# Imports
"""
from TempEnv import TempEnv as ENV
"""
from lstm import LSTM
from env import LSTMEnv as ENV
#"""
from Agent import GOD

if __name__=="__main__":
    """
    Making the actionSpace based on '-a = Number of possible actions

    Case 1: If the number of actions are even
    actionSpace = List of actions based on percentage change to be made on the current price 
                  with fixed difference of 5% between consecutive actions in the list

    Ex - For -a = 4 | actionSpace = [-10, -5, 5, 10]
         For -a = 8 | actionSpace = [-20, -15, -10, -5, 5, 10, 15, 20]

    Case 2: If the number of actions are odd
    actionSpace = List of actions based on percentage change to be made on the current price 
                  with fixed difference of 5.5% between consecutive actions in the list

    Ex - For -a = 5 | actionSpace = [-11.0, -5.5, 0.0, 5.5, 11.0]
         For -a = 9 | actionSpace = [-22.0, -16.5, -11.0, -5.5, 0.0, 5.5, 11.0, 16.5, 22.0]

    Note - The actions are centered around zero i.e the mean of actionSpace is zero in both cases
    #"""
    n=int(keywords['a']//2)
    if keywords['a']%2==0:
        actionSpace = [i for i in range(-n*5,n*5+1,5)]
        actionSpace.pop(len(actionSpace)//2)
    else:
        actionSpace = [i/10 for i in range(-n*55,n*55+1,55)]
    keywords['a']=actionSpace
    # If path for trained model is not given or fine-tuning is enabled then training process is initiated
    if keywords["p"] is None or keywords["f"]:
        print("Model Will be trained")
        print(f"Debuggin set to {keywords['d']}")
        log.info(f"Model will be trained")
        # Initializing GOD (master) agent 
        god = GOD(
            stateSize=int(stateSize),
            nAgent=int(keywords["n"]),
            maxEpisode=int(keywords["e"]),
            trajectoryLength=int(keywords["t"]),
            actionSpace=keywords["a"],
            alr=keywords["alr"],
            clr=keywords["clr"],
            debug=keywords["d"],
            path=keywords["p"],
        )

        # Logging all details about the God agent created during training phase
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
            # Train and save the trained model if no exceptions are encountered
            god.train()
            # Save model
            god.saveModel("../Saved_model")
        except KeyboardInterrupt:
            # Save model if interrupted by user interrupt
            god.saveModel("../Saved_model")
        except Exception as catch:
            #log.debug(f"Terminaion Trace back {catch.with_traceback()}")
            threadCount+=1
            # Log the exception details for debugging
            log.info(f"Terminated on a {threadCount}\t{catch}")
            log.info(f"Traceback for the {threadCount} Exception\t{sys.exc_info()}")
            print(f"{threadCount} thread Terminated, check log")
        print("Training Complete")
    else:
        # Testing the model
        print("Model will be tested")
        log.info(f"Model will be tested")
        print(f"Debuggin set to {keywords['d']}")
        # GOD (master) agent is initialized in debug mode for testing with trained model
        god = GOD(
            stateSize=int(stateSize),
            # Not using Boss agents while Online/Offline testing
            #nAgent=int(keywords["n"]),
            trajectoryLength=int(keywords["t"]),
            actionSpace=keywords["a"],
            path=keywords["p"],
            debug=keywords["d"],
            alr=keywords["alr"],
            clr=keywords["clr"],
        )

        # Logging all details about the God agent created during testing phase
        print("Master Agent Made")
        log.info("GOD inititated")
        log.info(f"State Size = {int(stateSize)}")
        log.info(f"Action Space = {keywords['a']}")
        log.info(f"Debug = {keywords['d']}")
        log.info(f"Path of Model = {keywords['p']}")
        log.info(f"Steps tested for = {keywords['s']}")

        # Some params used wrt to LSTM network
        output_size = 13
        input_dim = output_size
        hidden_dim = 40
        layer_dim = 2

        # LSTM model
        model = LSTM(output_size, input_dim, hidden_dim, layer_dim,debug=keywords["d"])
        # Loading saved model from the given path
        model.loadM("ENV_MODEL/lstm_modelV3.pt")

        # Generating environment using the loaded LSTM model
        env=ENV(
            model=model,
            dataset_path="../datasets/normalized_weird_13_columns_with_supply.csv",
            actionSpace=actionSpace,
            debug=keywords["d"],
        )
        print("Environment inititated")
        #env = ENV(stateSize,actionSpace)
        
        log.info("Environment inititated")
        god.giveEnvironment(env)
        log.info("Environment parsed, Boss inititated")
        log.info(f"ENV LSTM: {model}")

        # Testing phase
        # if testing timeSteps <=0 then proceed in online mode, else offline
        time=int(keywords['s'])
        god.test(time=time)

        print("Testing Complete")
