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

State = ((Market Demand)**2+(Ontario Demand)**2),Ontario Price,Northwest,Northeast,Ottawa,East,Toronto,Essa,Bruce, (TIMEstamp - optional)
"""
__author__ = 'BlackDChase,MR-TLL'
__version__ = '0.0.5'

# Imports
from torch import nn, multiprocessing, device, Tensor
from torch.distributions import Categorical
import torch
import numpy as np
from tqdm import tqdm
import threading
import log
import sys
from NeuralNet import Network

# GLOBAL
#device = device("cuda" if torch.cuda.is_available() else "cpu")
"""
if torch.cuda.is_available():
    log.info("Using Allmight")
else:
    log.info("Might missing")
#"""

torch.autograd.set_detect_anomaly(True)
"""
Better error logging for inplace operations that throw errors in automatic differentiation.

#"""

class GOD:
    '''
    This class is responsible for taking the final action which will be conveyed to the enviorenment
    and the management of BOSS agents which will explore the enviorenment.

    Also containing the initialiation of the policy Net and Critic Net.

    @Input (From Enviorenment) :: The Current State.
    @Output (To enviorenment)  :: The Final Action Taken.
    '''
    def __init__(self,maxEpisode=100,nAgent=1,debug=False,trajectoryLength=25,stateSize=9,actionSpace=11,name="GOD"):
        '''
        Initialization of various GOD parameters, self evident from the code.
        #'''
        self.name=name
        self.setMaxEpisode(maxEpisode)
        self.setNumberOfAgent(nAgent)
        self.setTrajectoryLength(trajectoryLength)
        self.__bossAgent = []
        self.debug = debug
        self.price = 0
        self.__actorLR = 1e-3
        self.__criticLR = 1e-3
        # state is the 9 dimentional tensor , defined at the top
        self.stateSize = stateSize
        self._state = Tensor([0]*stateSize)

        # action space is the percent change of the current price.
        self._actionS = actionSpace
        self.__makeActions()

        # semaphores do deal with simultaneous updates of the policy and critic net by multiple bosses.
        self._policySemaphore = threading.Semaphore()
        self._criticSemaphore = threading.Semaphore()

        ## defining the actual neural net itself, input is state output is probability for each action.
        self.__policyNet = Network(
            len(self._state),
            len(self._actionSpace),
            lr=self.__actorLR,
            L1=(nn.Linear,20,nn.Tanh()),
            L2=(nn.Linear,50,nn.Softmax(dim=0)),
            debug=self.debug,
            ## we will add softmax at end , which will give the probability distribution.
        )

        # Could be updated
        ## the critic network :: it's input is state and output is a scaler ( the value of the state)
        self.__criticNet =Network(
            len(self._state),
            1,
            lr=self.__criticLR,
            L1=(nn.Linear,30,nn.ReLU6()),
            L2=(nn.Linear,40,nn.ReLU6()),
            debug=self.debug,
        )
        #'''
        pass

    def __makeActions(self):
        actionSpace = [i/10 for i in range(-self._actionS*25,self._actionS*25+1,25)]
        self._actionSpace = Tensor(actionSpace)

    def giveEnvironment(self,env):
        self.__env=env
        self.__initateBoss()
        return

    def setNumberOfAgent(self,nAgent):
        self.__nAgent = nAgent
        return

    def getActionSpace(self):
        return self._actionSpace

    def setMaxEpisode(self,maxEpisode):
        self.maxEpisode = maxEpisode
        return

    def setTrajectoryLength(self,trajectoryLength):
        self.trajectoryLength = trajectoryLength
        return

    """
    Wont be needed as take action is called by enviornment and will provide current state
    def getState(self):
        # To be defined Later (Get the current state)
        pass
    #"""

    def train(self):
        self.__trainBoss()
        return

    def _updatePolicy(self,loss):
        loss.backward(retain_graph=True)
        self.__policyNet.optimizer.step()
        self.__policyNet.optimizer.zero_grad()
        return

    def _updateCritc(self,loss):
        loss.backward(retain_graph=True)
        self.__criticNet.optimizer.step()
        self.__criticNet.optimizer.zero_grad()
        return

    def takeAction(self,state):
        '''
        Take the final action according to the Policy Network.
        This method is not called inside GOD.
        This method is only called by ENV, every time it decides to take price for next time step.
        Enviorenment will send current state and this take action will return an action via env.step
        This will be done using pretrained policyNetwork.
        #'''
        actionProb = self._getAction(state)
        pd = Categorical(probs=actionProb)
        ## create a catagorical distribution acording to the actionProb
        ## categorical probability distribution
        actionIndex = pd.sample()
        probab = actionProb[actionIndex]
        nextState,reward,info = self.__env.step(actionIndex)
        log.info(f"{self.name} step taken,{info}, rewards = {reward}")
        if self.debug:
            log.debug(f"Expected next State: {nextState}")
        return actionIndex,probab

    def _peakAction(self,state,action):
        '''
        Online dilemma
        will be used at training time , for updating the networks
        #'''
        result = self.__env.step(state,action)
        return result

    def _getAction(self,state):
        self._policySemaphore.acquire()
        actionProbab = self.__policyNet.forward(state)
        # Not sure if forward is the way to go
        self._policySemaphore.release()
        return actionProbab

    def _getCriticValue(self,state):
        self._criticSemaphore.acquire()
        vVlaue = self.__criticNet.forward(state)
        self._criticSemaphore.release()
        return vVlaue

    def reset(self):
        return self.__env.reset()

    def step(self,action):
        return self.__env.step(action)

    def __initateBoss(self):
        '''
        Initialize all the boss agents for training
        '''
        for i in range(self.__nAgent):
            if len(self.__bossAgent)<i+1:
                self.__bossAgent.append(BOSS(
                    god=self,
                    name="BOSS "+str(i).zfill(2),
                    depth=200,
                    maxEpisode=self.maxEpisode,
                    debug=self.debug,
                    trajectoryLength=self.trajectoryLength,
                    stateSize=self.stateSize,
                ))
            else:
                self.__bossAgent[i]=BOSS(
                    god=self,
                    name="BOSS "+str(i),
                    depth=200,
                    maxEpisode=self.maxEpisode,
                    debug=self.debug,
                    trajectoryLength=self.trajectoryLength,
                    stateSize=self.stateSize,
                )
            if self.debug:
                log.debug(f"Boss{str(i).zfill(2)} created")
        return self.__bossAgent

    def __trainBoss(self):
        # To be defined Later :: the actual function to train multiple bosses.
        bossThreads=[]
        for i in range(self.__nAgent):
            process = multiprocessing.Process(target=self.__bossAgent[i].train)
            process.start()
            if self.debug:
                log.debug(f"Boss{str(i).zfill(2)} training started via GOD")
            bossThreads.append(process)
        for i in bossThreads:
            i.join()
        return

    pass

class BOSS(GOD):
    '''
   The actual class which does the exploration of the state space.
   Contains the code for the actor critic algorithm (Trajectory generation+ Policy gradient and value net updation )

   @Input :: (From Enviorenment) The current state + next state according to the current price to create trajectory.

   @Output:: (To Enviorenment)   The action taken for creating trajectory.

   @Actual Job :: To update the policy network and critic net by creating the trajectory and calculating losses.


    '''
    def __init__(self,
                maxEpisode,
                god,
                trajectoryLength,
                stateSize,
                name,
                gamma=0.99,
                # lamda=0.1, # Lambda was earlier used for GAE
                depth=200,
                debug=False,
                 ):
        super(BOSS,self).__init__(maxEpisode=maxEpisode,debug=debug,trajectoryLength=trajectoryLength,stateSize=stateSize,name=name)
        self.trajectoryS = torch.zeros([self.trajectoryLength,self.stateSize])
        self.trajectoryR = torch.zeros(self.trajectoryLength)
        self.trajectoryA = torch.zeros(self.trajectoryLength)
        self.god = god
        self.ɤ = gamma
        self.d = depth
        self.vPredicted = torch.zeros(self.trajectoryLength)
        self.vTarget = torch.zeros(self.trajectoryLength)
        self.advantage = torch.zeros(self.trajectoryLength)
        # If entropy H_t calculated, Init beta
        pass

    def train(self):
        '''
        The Actual function to train the network , the actor-critic actual logic.
        Eviorienment.step :: env.step has to give different outputs for different state trajectories by
        different boss. it has to take in account the diff trajectories becouse diff bosses will go to
        different states.
        #'''
        if self.debug:
            log.debug(f"{self.name} training started inside BOSS")

        # here the main logic of training of A2C will be present
        for e in tqdm(range(self.maxEpisode)):
            self.startState = self.god.reset()
            self.gatherAndStore()
            """ @BOSS
            Do we need to intiallise here?? when we are re declaring it in the three cal methods
            Also, if we are declaring them lets declare them instart, and keep it in device
            and After each gatherAndStore reset it.
            ## @BLACK :: NO we dont need to initialize here ,i have removed it.
            Also what which advantage function are we calling? @Black :: Nstep advantage.
            #"""
            self.calculateV_p()
            self.calculateV_tar()
            self.calculateNSTEPAdvantage()

            '''
            Question to be figured out :: Exactly when should the boss agents update the networks??
            #'''
            self.calculateAndUpdateL_P()
            # calculate  policy loss and update policy network
            self.calculateAndUpdateL_C()
            # calculate critic loss and update critic network
            log.info(f"{self.name} episode {e} Completed")
        pass


    def gatherAndStore(self):
        # gather a trajectory by acting in the enviornment using current policy
        ''' @BOSS
        Do we need any changes in here?
        @Black:i dont think so , trajectory is being gathered here step by step. no problems.
        Maybe tensorize the code??
        #'''
        currentState = self.startState
        log.info(f"Starting state={currentState}")
        for i in range(self.trajectoryLength):
            action,actionProb = self.getAction(currentState)
            #nextState,reward,info = self.god.step(currentState,action)
            nextState,reward,info = self.god.step(action)
            ## Oi generous env , please tell me the next state and reward for the action i have taken
            log.info(f"{self.name},  {info}")
            self.trajectoryS[i],self.trajectoryA[i],self.trajectoryR[i] = currentState,actionProb,reward
            if self.debug:
                log.debug(f"Action = {action}")
            currentState=nextState
        if self.debug:
            log.debug(f"Action Probability = {self.trajectoryA}")
            log.debug(f"rewards = {self.trajectoryR}")
        pass

    def getAction(self,state):
        '''
        Responsible for taking the correct action from the given state using the neural net.
        @input :: current state
        @output :: the index of action which must be taken from this states, and probability of that action
        #'''
        state = state.float()
        actionProb = self.god._getAction(state)
        ## This creates state-action probability vector from the policy net. 
        pd = Categorical(probs=actionProb) ## create a catagorical distribution acording to the actionProb
        ## categorical probability distribution
        actionIndex = pd.sample() ## sample the action according to the probability distribution.
        probab = actionProb[actionIndex]

        return actionIndex,probab

    def calculateV_p(self):
        # calculate the predicted v value by using critic network :: Predicted value is just the value returned by the critic network.
        self.vPredicted.zero_()
        # This resets the tensor to zero
        for i in range(self.trajectoryLength):
            state=self.trajectoryS[i]
            self.vPredicted[i]=self.god._getCriticValue(state)
        return

    def calculateV_tar(self):
        # calculate the target value v_tar using critic network
        '''
        This is a huge topic of debate , i.e how to actually calculate the target value, currently we have 2 propositions.
        1. v_target(s) = summation( reward + v_predicted(ss)) , where ss is some state after the trajectory.
        2. calculate v_target with the help of advantage function itself.

        #####################################################################################################
        Another Huge Doubt regarding v_target and GAE is::

        DO WE NEED TO CONSTRUCT NEW EXPERIENCE(ENVIORENMENT EXPLORATION) FOR CALCULATING V_tar OR WE CAN USE THE
        CURRENT TRAJECTORY FOR THIS PURPOSE(more probable)

        CHOICE 1 :: Use TD error , for each state in trajectory , use TD error to calculate V_tar since
        you know 𝛄 + next state reward and you can calculate value of the next and current state
        Pros :: very fast and TD error is a reliable and solid method.
        Cons :: Maybe if we only see one step ahead , the estimate will be less reliable.

        CHOICE 2 :: For this we will, for each state in the trajectory, calculate the advantage and V_tar
        using the previous method (by travelling to the end of the trajectory and accumulating rewards as
        given in jamboard slide 15) , the only difference is we start from the current state itself to the
        end of trajectory. (Or until a depth)

        We have chosen choice 2 for v_tar , by terating in reverse direction in the trajectory list.
        #'''
        # we have set γ to be 0.99 // see this sweet γ @BlackD , α , β , θ 
        ## here 𝛄 can be variable so, the length can be changed.
        #  ans=0.0
        #  for i in range(0,200):
        #      ans+=((self.ɤ)**(i+1))*self.trajectory[i][2]
        # ans+=(self.ɤ)**200*self.god._getCriticValue((self.trajectory[200][0])) ## multiply by the actual value of the 200th state.
        #  return ans

        self.vTarget.zero_()
        self.vTarget[self.trajectoryLength-1] = self.trajectoryR[self.trajectoryLength-1]
        ## only the reward recieved in the last state , we can also put it zero i think
        # guess will have to consult literature on this, diff shouldn't be substantial.
        for i in reversed(range(self.trajectoryLength-1)):
            # iterate in reverse order.
            self.vTarget[i] = self.trajectoryR[i] + self.ɤ*self.vTarget[i+1]
            # v_tar_currentState = reward + gamma* v_tar_nextState
        return

    def calculateGAE(self):
        # calculate the Advantage using the critic network
        # gae is put on hold at this time
        # To be declared at latter stage
        pass


    def calculateNSTEPAdvantage(self):
        """
        Calculate Advantage using TD error/N-STEP , logic similar to vTarget calculation
        #"""
        vPredLast = self.vPredicted[self.trajectoryLength-1]
        self.advantage.zero_()
        self.advantage[-1]=vPredLast
        for i in reversed(range(self.trajectoryLength-1)):
            self.advantage[i] = self.trajectoryR[i] + self.ɤ*self.advantage[i+1] - self.vPredicted[i]
        if self.debug:
            log.debug(f"Advantage = {self.advantage}")
        return

    def calculateAndUpdateL_P(self):
        ### Semaphore stuff for safe update of network by multiple bosses.
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
        #'''
        self.god._policySemaphore.acquire()

        actionProb = self.trajectoryA
        loss = -1*torch.sum(self.advantage*torch.log(actionProb))
        loss/=self.trajectoryLength
        log.info(f"policy loss = {loss}")
        self.god._updatePolicy(loss)

        self.god._policySemaphore.release()
        return

    def calculateAndUpdateL_C(self):
        self.god._criticSemaphore.acquire()

        loss = torch.sum(torch.pow(self.vPredicted-self.vTarget,2))
        loss/=self.trajectoryLength
        log.info(f"critic loss = {loss}")
        self.god._updateCritc(loss)

        self.god._criticSemaphore.release()
        return
