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

State = ((Market Demand)**@+(Ontario Demand)**2),Ontario Price,Northwest,Northeast,Ottawa,East,Toronto,Essa,Bruce, (TIMEstamp - optional)
"""
__author__ = 'BlackDChase'
__version__ = '0.0.1'

# Imports
from torch import nn, multiprocessing, device, Tensor
import numpy as np
from tqdm import tqdm
import threading

# GLOBAL
device = device("cuda" if args.cuda else "cpu")

class Network(nn.Module):
    global device
    def __init__(self,stateSize,actionSize,**kwargs):
        super(Network,self).__init__()
        self.inputSize = stateSize
        self.outputSize = actionSize
        layers = []
        keyWords = list(kwargs.keys())
        kwargs["stateSize"] = (nn.Linear,stateSize,nn.ReLU)
        keyWords.insert(0,"stateSize")
        i=0
        for i in range(len(keyWords)-1):
            l1=keyWords[i]
            l2=keyWords[i+1]
            layers.append(kwargs[l1[0]](kwargs[l1[1]],kwargs[l2[1]]))
            if len(l1)>=3:
                layers.append(kwargs[l1[2]]())
        l1=keyWords[len(keyWords)-1]
        layers.append(kwargs[l1[0]](kwargs[l1[1]],actionSize))
        if len(l1)>=3:
            layers.append(kwargs[l1[2]]())

        self.model = nn.Sequential(*layers)

    def forward(self,currentState):
        probabilityDistributionParameter = self.model(currentState)
        return probabilityDistributionParameter
    pass

class GOD:
    def __init__(self,maxEpisode=100,nAgent=1,debug=False,trajectoryLenght=25):
        self.name="GOD"
        self.setMaxEpisode(maxEpisode)
        self.setNumberOfAgent(nAgent)
        self.setTrajectoryLength(trajectoryLenght)
        self.bossAgent = []
        self.price = 0
        self.state = Tensor([0]*9)
        self.actionSpace = np.array([-12.5,-10,-7.5,-5,-2.5,0,2.5,5,7.5,10,12.5])
        
        self.policySemaphore = threading.Semaphore()
        self.criticSemaphore = threading.Semaphore()
        self.policyNet = Network(
            len(self.state),
            len(self.actionSpace),
            "L1":(nn.Linear,20,nn.Tanh)
            "L2":(nn.Linear,50,nn.SELU)
        )
                # Could be updated
        self.criticNet =Network(
            len(self.state),
            1,
            "L1":(nn.Linear,30,nn.ReLU)
            "L2":(nn.Linear,40,nn.ReLU)
        )
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
            self.bossAgent.append(BOSS(self))

        pass

    def trainBoss(self):
        # To be defined Later
        bossThreads=[]
        for i in range(self.nAgent):
            process = multiprocessing.Process(target=self.bossAgent[i].train,args=self.state.__deepcopy__)
            process.start()
            bossThreads.append(process)
        for i in bossThreads:
            i.join()
        pass

    pass

class BOSS(GOD):
    def __init__(self,god,actorLearningRate=0.01,criticLearningRate=0.01,gamma=0.99):
        super().__init__()
        self.name='BOSS'

        self.a_lr=actorLearningRate
        self.c_lr=criticLearningRate
        self.trajectory = []
        self.god = god
        
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
        for _ in range(self.maxEpisode):
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
        self.god.policySemaphore.acquire()
        # Do stuff
        self.god.policySemaphore.release()
        pass

    def calculateAndUpdateL_C(self):
        self.god.criticSemaphore.acquire()
        # Do stuff
        self.god.criticSemaphore.acquire()
        pass
