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
__author__ = 'BlackDChase,MR-TLL'
__version__ = '0.0.3'

# Imports
from torch import nn, multiprocessing, device, Tensor
from torch.distributions import Catagorical
import torch
import numpy as np
from tqdm import tqdm
import threading
import logModule.log as log

# GLOBAL
device = device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    global device
    def __init__(self,stateSize,actionSize,lr=1e-3,**kwargs):
        """
        ## All logic below are initializing the Network with the layers.
        Parameters:
        __________
        : stateSize     : Input Size
        : actionSize    : Output Size
        : **kwargs      : Keyword arguments for different layers should be L1,L2,L3..
                          (because its a dic and will change the oredring atuomatically)
                          Later we kan have it Ordered dic to be directly sent throught
                          but would make less sense
                          The touple has type of layer, it's input and activation funtion if it's there.
        #"""
        super(Network,self).__init__()
        self.inputSize = stateSize
        self.outputSize = actionSize
        self.learningRate = lr
        layers = []
        keyWords = list(kwargs.keys())
        kwargs["stateSize"] = (nn.Linear,stateSize,nn.ReLU)

        """
        Optimizer and loss function
        #"""


        """
        Input layer
        #"""
        keyWords.insert(0,"stateSize")
        i=0
        for i in range(len(keyWords)-1):
            l1=keyWords[i]
            l2=keyWords[i+1]
            layers.append(kwargs[l1[0]](kwargs[l1[1]],kwargs[l2[1]]))
            if len(l1)>=3:
                """
                For layers with activation function
                #"""
                layers.append(kwargs[l1[2]]())
        l1=keyWords[len(keyWords)-1]
        layers.append(kwargs[l1[0]](kwargs[l1[1]],actionSize))
        """
        Output Layer
        #"""
        if len(l1)==3:
            """
            For layers with activation function
            #"""
            layers.append(kwargs[l1[2]]())

        self.model = nn.Sequential(*layers)

    def forward(self,currentState):
        """
        Override for forward from nn.Module
        To calculate the pd-parameters by using the neural net.
        __________
        Output for Policy network   : Probability Distribution Parameter
        Output for Critic network   : Advantage
        #"""
        output = self.model(currentState)
        return output
    pass

class GOD:
    '''
    This class is responsible for taking the final action which will be conveyed to the enviorenment
    and the management of BOSS agents which will explore the enviorenment.

    Also containing the initialiation of the policy Net and Critic Net.

    @Input (From Enviorenment) :: The Current State.
    @Output (To enviorenment)  :: The Final Action Taken.

    '''
    def __init__(self,env=env,maxEpisode=100,nAgent=1,debug=False,trajectoryLenght=25):
        '''
        Initialization of various GOD parameters, self evident from the code.
        '''
        self.name="GOD"
        self.setMaxEpisode(maxEpisode)
        self.setNumberOfAgent(nAgent)
        self.setTrajectoryLength(trajectoryLenght)
        self.__bossAgent = []
        self.price = 0
        self.__env = env
        self.__actorLR = 1e-3
        self.__criticLR = 1e-3
        # state is the 9 dimentional tensor , defined at the top
        self._state = Tensor([0]*9)

        # action space is the percent change of the current price.
        self._actionSpace = np.array([-12.5,-10,-7.5,-5,-2.5,0,2.5,5,7.5,10,12.5])

        # semaphores do deal with simultaneous updates of the policy and critic net by multiple bosses.
        self.__policySemaphore = threading.Semaphore()
        self.__criticSemaphore = threading.Semaphore()

        ## defining the actual neural net itself, input is state output is probability for each action.
        self.__policyNet = Network(
            len(self._state),
            len(self._actionSpace),
            lr=self.__actorLR,
            L1=(nn.Linear,20,nn.Tanh),
            L2=(nn.Linear,50,nn.Softmax), ## we will add softmax at end , which will give the probability distribution.
        )

        # Could be updated
        ## the critic network :: it's input is state and output is a scaler ( the value of the state)
        self.__criticNet =Network(
            len(self._state),
            1,
            lr=self.__criticLR,
            L1=(nn.Linear,30,nn.ReLU),
            L2=(nn.Linear,40,nn.ReLU),
        )
        #'''
        self.__initateBoss()
        pass

    def setNumberOfAgent(self,nAgent):
        self.__nAgent = nAgent
        pass

    def setMaxEpisode(self,maxEpisode):
        self.maxEpisode = maxEpisode
        pass

    def setTrajectoryLength(self,trajectoryLenght):
        self.trajectoryLength = trajectoryLenght
        pass

    def getState(self):
        # To be defined Later (Get the current state)
        pass

    def train(self):
        self.__trainBoss()
        pass

    def takeAction(self):
        '''
        Take the final action according to the Policy Network.
        '''
        pass

    def _peakAction(self,state,action):
        '''
        will be used at training time , for updating the networks
        '''
        result = self.__env.step(state,action)
        return result

    def _getAction(self,state):
        self.__policySemaphore.acquire()
        actionProbab = self.__policyNet.forward(state)
        # Not sure if forward is the way to go
        self.__policySemaphore.release()
        return actionProbab

    def _getCriticValue(self,state):
        self.__criticSemaphore.acquire()
        vVlaue = self.__criticNet.forward(state)
        self.__criticSemaphore.release()
        return vVlaue

    def __initateBoss(self):
        '''
        Initialize all the boss agents for training
        '''
        for _ in range(self.__nAgent):
            self.__bossAgent.append(BOSS(self,depth=200))

        pass

    def __trainBoss(self):
        # To be defined Later :: the actual function to train multiple bosses.
        bossThreads=[]
        for i in range(self.__nAgent):
            process = multiprocessing.Process(target=self.__bossAgent[i].train)
            process.start()
            bossThreads.append(process)
        for i in bossThreads:
            i.join()
        pass

    pass

class BOSS(GOD):
    '''
   The actual class which does the exploration of the state space.
   Contains the code for the actor critic algorithm (Trajectory generation+ Policy gradient and value net updation )

   @Input :: (From Enviorenment) The current state + next state according to the current price to create trajectory.

   @Output:: (To Enviorenment)   The action taken for creating trajectory.

   @Actual Job :: To update the policy network and critic net by creating the trajectory and calculating losses.


    '''
    def __init__(self,god,gamma=0.99,depth=200,lamda=0.1):
        super().__init__()
        self.name='BOSS'
        self.trajectory = []
        self.god = god
        self.ɤ = gamma
        self.d = depth
        # If entropy H_t calculated, Init beta
        pass

    def train(self):
        '''
        The Actual function to train the network , the actor-critic actual logic.
        Eviorienment.step :: env.step has to give different outputs for different state trajectories by different boss. it has to take in account the diff
        trajectories becouse diff bosses will go to different states.

        '''
        # here the main logic of training of A2C will be present
        for _ in range(self.maxEpisode):
            # To be initialised
            vPredicted = torch.Tensor([0]*len(self.trajectoryLength))
            vTarget = torch.Tensor([0]*len(self.trajectoryLength))
            advantage = torch.Tensor([0]*len(self.trajectoryLength))
            self.startState = self.god.env.reset()
            self.gatherAndStore()
            """
            Wouldnt all these funtions below need `i` in some sense?
            UPDATE!! @ The below function are already calculating the summed values for the given trajectory,
            i believe this upper loop is unnecessary and must be removed!😄
            #"""
            vPredicted = self.calculateV_p()
            vTarget = self.calculateV_tar()
            advantage = self.calculateGAE()
            '''
            Question to be figured out :: Exactly when should the boss agents update the networks??
            '''
            self.calculateAndUpdateL_P(advantage)  # calculate  policy loss and update policy network
            self.calculateAndUpdateL_C(vPredicted,vTarget) # calculate critic loss and update critic network
        pass


    def gatherAndStore(self,initialState):
        # gather a trajectory by acting in the enviornment using current policy
        '''
        Incomplete
        #'''

        currentState=initialState
        for _ in self.trajectoryLength:
            action = self.getAction(currentState)
            nextState,reward,info = self.god.step(currentState,action)
            ## Oi generous env , please tell me the next state and reward for the action i have taken

            self.trajectory.append([currentState,action,reward])
            currentState=nextState
        return

    def getAction(self,state):
        '''
        Responsible for taking the correct action from the given state using the neural net.
        @input :: current state
        @output :: the action which must be taken from this states
        #'''
        state = torch.from_numpy(state.float())
        actionProb = self.god.getAction(state)
        ## This creates state-action probability vector from the policy net. 
        pd = Catagorical(logit=actionProb) ## create a catagorical distribution acording to the actionProb
        ## categorical probability distribution
        action = pd.sample() ## sample the action according to the probability distribution.
        # What does these 3 lines do??

        return action

    def calculateV_p(self,state):
        # calculate the predicted v value by using critic network :: Predicted value is just the value returned by the critic network.
        vValue = self.god._getCriticValue(state)
        return vValue

    def calculateV_tar(self):
        # calculate the target value v_tar using critic network 
        '''
        This is a huge topic of debate , i.e how to actually calculate the target value, currently we have 2 propositions.
        1. v_target(s) = summation( reward + v_predicted(ss)) , where ss is some state after the trajectory.
        2. calculate v_target with the help of advantage function itself.

        #################################################################################################################
        Another Huge Doubt regarding v_target and GAE is::

        DO WE NEED TO CONSTRUCT NEW EXPERIENCE(ENVIORENMENT EXPLORATION) FOR CALCULATING V_tar OR WE CAN USE THE
        CURRENT TRAJECTORY FOR THIS PURPOSE(more probable)

        CHOICE 1 :: Use TD error , for each state in trajectory , use TD error to calculate V_tar since
        you know 𝛄 + next state reward and you can calculate value of the next and current state
        Pros :: very fast and TD error is a reliable and solid method.
        Cons :: Maybe if we only see one step ahead , the estimate will be less reliable.

        CHOICE 2 :: For this we will, for each state in the trajectory , calculate the advantage and V_tar using the previous
        method(by travelling to the end of the trajectory and accumulating rewards as given in jamboard slide 15) , the only
        difference is we start from the current state itself to the end of trajectory. (Or until a depth)

        We have chosen choice 2 for v_tar , by terating in reverse direction in the trajectory list.
        '''
        # we have set γ to be 0.99 // see this sweet γ @BlackD , α , β , θ ( this is all tex , emacs master race , Ɣ ❈)
        ## here 𝛄 can be variable so, the length can be changed.
        ans=0.0
        for i in range(0,200):
            ans+=((self.ɤ)**(i+1))*self.trajectory[i][2]

        ans+=(self.ɤ)**200*self.god._getCriticValue((self.trajectory[200][0])) ## multiply by the actual value of the 200th state.
        return ans

    def calculateGAE(self):
        # calculate the Advantage using the critic network
        advantage=0
        return advantage

    def calculateTDAdvantage(self):
        ## Calculate Advantage using TD error
        pass

    def calculateAndUpdateL_P(self):    ### Semaphore stuff for safe update of network by multiple bosses.
        '''
        FOR UPDATING THE ACTOR USING POLICY GRADIENT WE MUST CALCULATE THE LOG PROBABILITY OF ACTION GIVEN
        A PARTICULAR STATE.BUT IN SITUATION OF MULTIPLE AGENTS IT MAY HAPPEN THAT BEFOR AGENT 1 FETCHES THIS
        PROBABILITY AND UPDATES AND UPDATES THE NETWORK , AGENT-2 MAY HAVE TAMPERED/UPDATED THE NETWORK.

        TO WORK AROUND THIS 2 CHOICES::
        CHOICE 1 :: KEEP TRACK OF THE STATE ACTION PROBABILITY FOR EACH AGENT'S TRAJECTORY , SO EVEN IF
        ANOTHER AGEND UPDATES THE NET, WE HAVE THE EXACT ORIGINAL PROBABILITY FROM WHEN THE AGENT HAD SAMPLED
        THE ACTION.

        CHOICE 2 :: BE IGNORANT , THE CHANCE THAT 2 AGENT HAVE TAMPERED WITH THE SAME STATE BEFOR UPDATING
        THE NETWORK WITH THEIR OWN LOSS IS EXTREMELEY LOW, SO IT DOESN'T MATTERS.
        '''
        self.god.policySemaphore.acquire()
        # Do stuff
        self.god.policySemaphore.release()
        pass

    def calculateAndUpdateL_C(self):
        self.god.criticSemaphore.acquire()
        # Do stuff
        self.god.criticSemaphore.acquire()
        pass
