from TempEnv import TempEnv as ENV
from Agent import GOD
import log

if __name__=="__main__":
    stateSize = 9
    log.debug(f"stateSize = {stateSize}")
    god = GOD(maxEpisode=100,nAgent=1,debug=True,trajectoryLength=25,stateSize=stateSize,actionSpace=11)
    log.info("GOD inititated")
    actionSpace = god.getActionSpace()
    log.info(f"Action space: {actionSpace}")
    env = ENV(stateSize,actionSpace)
    log.info("Environment inititated")
    god.giveEnvironment(env)
    log.info("Environment parsed, Boss inititated")
    try:
        god.train()
    except Exception as catch:
        log.debug(f"Terminated due to {catch}")
        print("Terminated, check log ",log.name)
