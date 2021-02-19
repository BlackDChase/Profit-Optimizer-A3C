"""
GOD AGENT
- Will take final action
- A supervising psuedo agent that takes action
- Definations required:
    - state
    - action
    - policy/critic Net
BOSS AGENT
- Will Update the network

((Market Demand)**@+(Ontario Demand)**2),Ontario Price,Northwest,Northeast,Ottawa,East,Toronto,Essa,Bruce
"""
__author__ = 'BlackDChase'
__version__ = '0.0.1'

import torch as pt

class GOD:
    def __init__(self,maxEpisode=100,nAgent=1,debug=False,trajectoryLenght=25):
        self.name="GOD"
        self.setMaxEpisode(maxEpisode)
        self.setNumberOfAgent(nAgent)
        self.setTrajectoryLength(trajectoryLenght)
        self.bossAgent = []
        self.price = 0
        self.state = pt.Tensor([0]*9) 
        # To be defined Later
        '''
        self.policyNet =
        self.criticNet =
        #'''
        pass

    def setNumberOfAgent(self,nAgent):
        self.nAgent = nAgent
        pass

    def setMaxEpisode(self,maxEpisode):
        self.maxEpisode = maxEpisode
        pass

    def setTrajectoryLength(self,trajectoryLenght):
        self.trajectoryLength = trajectoryLenght
        pass

    def getState(self):
        # To be defined Later
        self.updateBOSS()
        pass

    def updateBOSS(self):


    def takeAction(self):
        # To be defined Later
        pass

    def initateBoss(self):
        # To be defined Later
        for i in range(self.nAgent):
            self.bossAgent.append(BOSS())

        pass

    def trainBoss(self):
        # To be defined Later
        for i in range(self.nAgent):
            self.bossAgent[i].train(self.state.copy())
        pass

    pass

class BOSS(GOD):
    def __init__(self,actorLearningRate=0.01,criticLearningRate=0.01):
        super().__init__()
        self.name='BOSS'
        self.a_lr=actorLearningRate
        self.c_lr=criticLearningRate
        self.trajectory = []
        # If entropy H_t calculated, Init beta
        '''
        # To be initialised
        self.v_val_pred =
        self.v_val_target =
        self.advantage =
        '''

        pass

    def train(self,state):
        self.state=state
        # here the main logic of training of A2C will be present
        self.gatherAndStore()
        for i in self.trajectory:
            """
            Wouldnt all these funtions below need `i` in some sense?
            #"""
            self.v_val_pred += self.calculateV_p()
            self.v_val_target += self.calculateV_tar()
            self.advantage = self.calculateGAE()
        self.calculateAndUpdateL_P()  # calculate  policy loss and update policy network
        self.calculateAndUpdateL_C() # calculate critic loss and update critic network
        pass


    def gatherAndStore(self):
        # gather a trajectory by acting in the enviornment using current policy
        '''
        May be remove first element of the trajectory once that time has passed...
        And then add a new element until len(self.trajectory)<=self.trajectoryLenght
        #'''
        self.trajectory.append()
        pass

    def calculateV_p(self):
        # calculate the predictied v value by using critic network
        pass

    def calculateV_tar(self):
        # calculate the target value v_tar using critic network
        pass

    def calculateGAE(self):
        # calculate the Advantage uisng the critic network
        pass

    def calculateAndUpdateL_P(self):
        pass

    def calculateAndUpdateL_C(self):
        pass
