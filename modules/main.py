from TempEnv import TempEnv as ENV
from Agent import GOD
import log

if __name__=="__main__":
    stateSize = 9
    log.debug(f"stateSize = {stateSize}")
    god = GOD(
        maxEpisode=100,
        nAgent=1,
        debug=True,
        trajectoryLength=25,
        stateSize=stateSize,
        actionSpaceDeviation=5,
        # This means 5*2+1 number of action [-5*2.5, ... 0 , +5*2.5] percent change in price
    )
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
        #log.debug(f"Terminaion Trace back {catch.with_traceback()}")
        log.debug(f"Terminated due to {catch}")
        print("Terminated, check log ",log.name)
        raise catch
