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

State = Ontario Price, Ontario Demand, Ontario Supply,Northwest,Northeast,Ottawa,East,Toronto,Essa,Bruce, Northwest Nigiria, West
"""
__author__ = 'BlackDChase,MR-TLL'
__version__ = '1.0.0'

# Imports
from torch import nn, Tensor
from torch.distributions import Categorical
from collections import deque
import torch
import numpy as np
from torch.nn.modules.activation import SELU
from tqdm import tqdm
import log
import sys
from NeuralNet import Network #, Aktor, Kritc
import multiprocessing
from multiprocessing import Process #, Lock
multiprocessing.set_start_method('fork')
"""
from TempEnv import TempEnv as ENV
"""
from env import LSTMEnv as ENV
from lstm import LSTM

output_size = 13
input_dim = output_size
hidden_dim = 40
layer_dim = 2
envDATA="../datasets/normalized_weird_13_columns_with_supply.csv"
# ENV(LSTM,envDATA,actionSpace)
#"""


# GLOBAL
#device = device("cuda" if torch.cuda.is_available() else "cpu")
"""
if torch.cuda.is_available():
    log.info("Using Allmight")
else:
    log.info("Might missing")
#"""

#torch.autograd.set_detect_anomaly(True)
"""
Better error logging for inplace operations that throw errors in automatic differentiation.
#"""


class GOD:
    """
    This class is responsible for taking the final action which will be conveyed to the enviorenment
    and the management of BOSS agents which will explore the enviorenment.

    Also containing the initialiation of the policy Net and Critic Net.

    @Input (From Enviorenment) :: The Current State.
    @Output (To enviorenment)  :: The Final Action Taken.
    """
    def __init__(self,maxEpisode=100,nAgent=1,debug=False,trajectoryLength=25,stateSize=9,actionSpace=[-15,-10,0,+10,+15],name="GOD",path=None,alr=1e-3,clr=1e-3):
        """
        Initialization of various GOD parameters, self evident from the code.
        #"""
        self.name=name

        # We will be interacting with the env whose state will have 13 dimensions
        # i.e for every timeStep the env will represent its current state as a tuple of 13 values 
        # Dimensions of the state = 13 
        self.stateSize = stateSize

        # state is the 13 dimentional tensor, defined at the top
        self._state = Tensor([0]*stateSize)
        # action space is the list of percent change allowed on the current price based on the number of actions.
        self._actionSpace = Tensor(actionSpace)
        # Enable debugging if true 
        self.debug = debug

        self.price = 0
        self.__nAgent=nAgent
        self.__bossAgent=[]

        # Initializing actor and critic network
        self.makeNetwork(maxEpisode,nAgent,trajectoryLength,alr,clr)

        # Load model if path is specific or else raise exception error if the path is invalid 
        # If model doesnt exist in that path, it will randomly initialize a model which could be trained later
        if self.__class__==GOD:
            try:
                if not (path=="None" or path==None):
                    self.__loadModel(path)
                print("Model loaded from : ",path)
            except:
                print("Model Not Found")
        pass

    def makeNetwork(self,maxEpisode,nAgent,trajectoryLength,alr,clr):
        """
        Initializes the actor and critic network based on the specified params
        """

        # Setting up relevant params for actor and critic networks using defined setters 
        self.setMaxEpisode(maxEpisode)
        self.setNumberOfAgent(nAgent)
        self.setTrajectoryLength(trajectoryLength)
        self.__actorLR = alr
        self.__criticLR = clr

        """
        Semaphores do deal with simultaneous updates of the policy and critic net by multiple bosses.
        Not required, as model works without it.
        """
        """
        self._policySemaphore = Lock()
        self._criticSemaphore = Lock()
        self._resetSemaphore = Lock()
        #"""
        #"""
        ## defining the actual neural net itself, input is state output is probability for each action.
        self.__policyNet = Network(
            self.stateSize,
            len(self._actionSpace),
            lr=self.__actorLR,
            name="Policy Net",
            L1=(nn.Linear,20,nn.SELU()),
            #L2=(nn.Linear,18,nn.Softmax(dim=0),nn.Dropout(p=0.3)),
            L2=(nn.Linear,30,nn.Softmax(dim=0)),
            debug=self.debug,
            ## we will add softmax at end , which will give the probability distribution.
        )

        # Could be updated
        ## the critic network :: it's input is state and output is a scaler ( the value of the state)

        self.__criticNet = Network(
            self.stateSize,
            1,
            lr=self.__criticLR,
            name="Critic Net",
            L1=(nn.Linear,15,nn.SELU()),
            L2=(nn.Linear,10,nn.Tanh()),
            debug=self.debug,
        )
        """
        self.__policyNet = Aktor()
        self.__criticNet = Kritc()
        #"""

        """
        After using this, we might not need to have a semaphore when we use multiprocessesing
        [Refer](https://pytorch.org/docs/stable/notes/multiprocessing.html)
        To be sent to device before sharing
        #"""
        """
        self.__policyNet.share_memory()
        self.__criticNet.share_memory()
        #"""
        log.info(f"Policy Network: {self.__policyNet}")
        log.info(f"Critic Network: {self.__criticNet}")
        return

    # SETTERS -----------------------------------------------------

    def giveEnvironment(self,env):
        self._env=env
        return

    def setNumberOfAgent(self,nAgent):
        self.__nAgent = nAgent
        return
    
    def setMaxEpisode(self,maxEpisode):
        self.maxEpisode = maxEpisode
        return

    def setTrajectoryLength(self,trajectoryLength):
        self.trajectoryLength = trajectoryLength
        return

    # GETTERS -----------------------------------------------------
    
    def getActionSpace(self):
        return self._actionSpace

    # -------------------------------------------------------------

    # The master agent will train with the environment using the worker (BOSS) agents
    def train(self):
        log.info("This GOD will train with the enviornment")
        
        # Initialize (BOSS) worker agents 
        self.__initateBoss()
        # Train using the worker agents 
        self.__trainBoss()
        return
    
    # TODO, Update this
    def test(self,time=100):
        '''
        onlineUpdateLength is the amount of episodes the online agent must wait to gather
        trajectory from the real env, after which it finally updates the network.
        MODIFIED::
        According to the discussions at 10th Mat 8 PM , onlineUpdateLength is just the
        trajectory length for online agent. It won't wait to update the net , maybe except
        initially.
        #'''

        # Run in offline mode if testing timeSteps > 0 else in online mode
        if time>0:
            self.__offline(time)
        else:
            self.__online()
        return

    # TODO, Update this
    def __offline(self,time):
        a3cState = self.__getA3Cstate(time)
        normalStates = self.__getNormalStates(time=time)
        a3cProfit,normalProfit,diff = self.__compare(a3cState=a3cState,normalState=normalStates)
        return diff

    def __getA3Cstate(self,time):
        """
        @input      : Number of time steps for which this model is going to be tested
        @output     : Returns the `time` number of states which occured on the basis of Agent's response.
        #"""

        # Get the initial state from the environment
        currentState = self.reset()
        a3cState=[]
        for i in range(time):
            if self.debug:
                log.debug(f"a3cState={currentState}")
            
            a3cState.append(currentState)
            # Decide what action to take from currentState
            action,_ = self.decideAction(torch.Tensor(currentState))
            # Observe nextState, reward, info from the environment after taking the action 
            nextState,reward,_,info = self.step(action)
            ## Oi generous env , please tell me the next state and reward for the action i have taken
            log.info(f"{self.name}, {i},  {info}")
            if self.debug:
                log.debug(f"Reward and Shape = {reward}, {reward.shape}")
                log.debug(f"Action for {self.name} {i} = {action}, {type(action)}")
            # Change the currentState to nextState after action has been taken
            currentState=torch.Tensor(nextState)

        a3cState.append(currentState)
        a3cState = torch.stack(a3cState)
        return a3cState

    # Returns a trajectory of states using only LSTM without A3C feedback
    def __getNormalStates(self,time=100):
        normalState = Tensor(self._env.possibleState(time))
        return normalState

    # Calculates profits in price with and without A3C (only using LSTM) and their difference for comparison purposes
    def __compare(self,a3cState,normalState):
        """
        @input          : a3cState,time,normalState
        a3cState        : Output of states with a3C's feedback
        normalState     : Output of states without a3c's feedback
        time            : Timesteps for which this model in being tested
        #"""
        normalProfit=[]
        a3cProfit=[]

        # Indices of attributes based on the dataset we have considered
        supply_index = 2
        demand_index = 1
        price_index = 0
        
        # Computing profits from the list of states
        for i in range(len(a3cState)):
            # Profits generated from using only LSTM network
            normalProfit.append(normalState[i][price_index]*(normalState[i][demand_index]-normalState[i][supply_index]))
            # Profits generated using A3C
            a3cProfit.append(a3cState[i][price_index]*(a3cState[i][demand_index]-a3cState[i][supply_index]))

        diff=[]
        # Computing diff in normal and A3C profits and appending them to diff list
        for i in range(len(a3cProfit)):
            diff.append(a3cProfit[i]-normalProfit[i])
        return a3cProfit,normalProfit,diff
    
    def _updatePolicy(self,lossP):
        self.__policyNet.optimizer.zero_grad()
        lossP.backward(retain_graph=True)
        self.__policyNet.optimizer.step()
        return

    def _updateCritc(self,lossC):
        self.__criticNet.optimizer.zero_grad()
        lossC.backward(retain_graph=True)
        self.__criticNet.optimizer.step()
        return

    # TODO, Update this
    def decideAction(self,state):
        """
        Responsible for taking the correct action from the given state using the neural net.
        @input :: current state
        @output :: the index of action which must be taken from this states, and probability of that action
        #"""
        state = state.float()
        actionProb = self._getAction(state)
        """
        actionProb  ::  State-action probability vector from the policy net.
        Categorical ::  Create a categorical distribution acording to the actionProb
        #"""
        probabDistribution = Categorical(probs=actionProb)

        if self.debug:
            log.debug(f"Deciding action for {state}")
            log.debug(f"Probability Distribution {probabDistribution}, actionProbab = {actionProb}")
        
        try:
            # Sampling from the probDistribution if no exceptions occur
            actionIndex = probabDistribution.sample()
        except RuntimeError:
            # For invalid multinomial distribution (encountering probability entry < 0)
            actionIndex = np.random.randint(0,len(self._actionSpace))
        ## sample the action according to the probability distribution.
        if self.debug:
            log.debug(f"Action: {actionIndex}")
        return actionIndex,probabDistribution

    def _getAction(self,state):
        #self._policySemaphore.acquire()
        actionProbab = self.__policyNet.forward(state)
        #self._policySemaphore.release()
        return actionProbab

    def _getCriticValue(self,state):
        #self._criticSemaphore.acquire()
        vVlaue = self.__criticNet.forward(state)
        #self._criticSemaphore.release()
        return vVlaue

    # Returns an initial state from the env (Randomized)
    def reset(self):
        return torch.Tensor(self._env.reset())

    # Take the specific action on the env and returns nextState, reward and other relevant params
    def step(self,action):
        return self._env.step(action)

    def __initateBoss(self):
        """
        Initialize all the boss agents for training
        #"""
        self.__bossAgent=[]
        for i in range(self.__nAgent):
            boss = BOSS(
                god=self,
                name="BOSS "+str(i).zfill(2),
                maxEpisode=self.maxEpisode,
                debug=self.debug,
                trajectoryLength=self.trajectoryLength,
                stateSize=self.stateSize,
            )
            self.__bossAgent.insert(i,boss)
            if self.debug:
                log.debug(f"Boss{str(i).zfill(2)} created")
        return self.__bossAgent

    def __trainBoss(self):
        # To be defined Later :: the actual function to train multiple bosses.
        #"""
        bossThreads=[]
        for i in range(self.__nAgent):
            process = Process(
                target=self.__bossAgent[i].train,
                #args=(self._resetSemaphore,),  # Not using semaphores
            )
            bossThreads.append(process)

        for i in range(len(bossThreads)):
            bossThreads[i].start()
            log.info(f"Boss{str(i).zfill(2)} training started via GOD")
        for i in bossThreads:
            i.join()

        return
 
    def forwardP(self,actions):
        """
        To make forward pass in the policy network using the state trajectory
        @input          = states (list of tuples of states acc to trajectory followed)
        @output         = list showing probability distribution of possible actions 
        """

        #self._policySemaphore.acquire()
        actionProb = self.__policyNet.forward(actions)
        #self._policySemaphore.release()
        return actionProb

    # TODO, Update this
    def __online(self):
        self.gamma=0.9 ## gamma wasn't defined in GOD
        episodes=0
        # These self variables necessary for doing gather and store.
        # The same way they are used in the BOSS agent.
        # initialized in resetTrajectory method.
        self.resetTrajectory()
        currentState=self.reset()
        # get start state from the env.
        
        try:
            for _ in range(self.trajectoryLength):
                currentState=self.onlineGatherAndStore(currentState)
            while(True):
                ## remove the first entry
                #print("Episodes: ",episodes)
                self.trajectoryS.popleft()
                self.trajectoryR.popleft()
                self.trajectoryA.popleft()

                currentState=self.onlineGatherAndStore(currentState)
                self.calculateV_tar()
                self.calculateV_p()
                self.onlineNstepAdvantage()

                self.calculateAndUpdateL_P()
                self.calculateAndUpdateL_C()
        except KeyboardInterrupt:
            print("Online training terminated >_<")
            print("Episodes: ",episodes)
            # TODO, dont you wanna save or something?
        return

    def resetTrajectory(self):
        self.trajectoryS =  deque([], maxlen=self.trajectoryLength) #[[0 for i in range(self.stateSize)] for j in range(self.trajectoryLength) ] # torch.zeros([self.trajectoryLength,self.stateSize])
        self.trajectoryR =  deque([], maxlen=self.trajectoryLength) # torch.zeros(self.trajectoryLength)
        self.trajectoryA =  deque([], maxlen=self.trajectoryLength)
        self.vPredicted  = torch.zeros(self.trajectoryLength)#[0 for i in range(self.trajectoryLength)]
        self.vTarget     = torch.zeros(self.trajectoryLength)#[0 for i in range(self.trajectoryLength)]
        self.advantage   = torch.zeros(self.trajectoryLength)#[0 for i in range(self.trajectoryLength)]
        return

    # TODO, Update this
    def onlineGatherAndStore(self,currentState):
         action,probab = self.decideAction(currentState)
         #nextState,reward,info = self.god.step(action)
         nextState,reward,_,info = self.step(action)
         log.info(f"Online: {self.name}, {info}")
         if self.debug:
            log.debug(f"Online: Reward and Shape = {reward}, {reward.shape}")
         print(len(self.trajectoryS))
         self.trajectoryS.append(currentState)
         self.trajectoryA.append(action)
         self.trajectoryR.append(reward) #torch.Tensor(reward).tolist()
         if self.debug:
            log.debug(f"Online: Action for {self.name}, {action}, {type(action)}")
            log.debug(f"Online: Detached Next state {nextState}")
            currentState=torch.Tensor(nextState)
         if self.debug:
            log.debug(f"Online: Action = {self.trajectoryA}")
            log.debug(f"Online: vPred = {self.vPredicted}")
            log.debug(f"Online: vTar = {self.vTarget}")
         return nextState

    def calculateV_p(self):
        # calculate the predicted v value by using critic network
        self.vPredicted = Tensor(len(self.vPredicted))
        for i in range(self.trajectoryLength):
            state=torch.Tensor(self.trajectoryS[i])
            self.vPredicted[i]=self._getCriticValue(state)
        return

    def calculateV_tar(self):
        """
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

        We have chosen choice 2 for v_tar , by iterating in reverse direction in the trajectory list.
        #"""
        self.vTarget = Tensor(len(self.vTarget))
        
        self.vTarget[self.trajectoryLength-1] = (self.trajectoryR[self.trajectoryLength-1])
        for i in reversed(range(self.trajectoryLength-1)):
            # iterate in reverse order.
            self.vTarget[i] = (self.trajectoryR[i]) + self.gamma*self.vTarget[i+1]
            # v_tar_currentState = reward + gamma* v_tar_nextState
        return

    def onlineNstepAdvantage(self):
        """
        Calculate Advantage using TD error/N-STEP for online scenario.
        """
        vPredLast = self.vPredicted[self.trajectoryLength-1]
        self.advantage =  Tensor(len(self.advantage))
        self.advantage[-1]=vPredLast
        for i in reversed(range(self.trajectoryLength-1)):
            self.advantage[i] = torch.Tensor(self.trajectoryR[i]) + self.gamma*self.advantage[i+1] - self.vPredicted[i]
        log.info(f"Online : Advantage {self.name} = {self.advantage}")
        return

    def calculateAndUpdateL_P(self):
        """
        FOR UPDATING THE ACTOR USING POLICY GRADIENT WE MUST CALCULATE THE LOG PROBABILITY OF ACTION GIVEN
        A PARTICULAR STATE.BUT IN SITUATION OF MULTIPLE AGENTS IT MAY HAPPEN THAT BEFOR AGENT 1 FETCHES THIS
        PROBABILITY AND UPDATES AND UPDATES THE NETWORK , AGENT-2 MAY HAVE TAMPERED/UPDATED THE NETWORK.

        TO WORK AROUND THIS 2 CHOICES::
        CHOICE 1 :: KEEP TRACK OF THE STATE ACTION PROBABILITY FOR EACH AGENT'S TRAJECTORY , SO EVEN IF
        ANOTHER AGEND UPDATES THE NET, WE HAVE THE EXACT ORIGINAL PROBABILITY FROM WHEN THE AGENT HAD SAMPLED
        THE ACTION.

        CHOICE 2 :: BE IGNORANT , THE CHANCE THAT 2 AGENT HAVE TAMPERED WITH THE SAME STATE BEFOR UPDATING
        THE NETWORK WITH THEIR OWN LOSS IS EXTREMELEY LOW, SO IT DOESN'T MATTERS.

        # Advantage detached because: Advantage is the result of critc, and if it is backpropagated here,
        critic model might face issues during it's own backpropagation
        #"""
        pd = self.auxillaryForward(torch.Tensor(self.trajectoryS))
        
        dist = Categorical(pd)
        logProb = dist.log_prob(torch.Tensor(list(self.trajectoryA)))

        advantage = self.advantage.detach()
        if self.debug:
            log.debug(f"Online: advantage detached for {self.name}")
        loss = -1*torch.mean(advantage*logProb)
        #log.info(f"Policy loss = {loss}")
        self._updatePolicy(loss)
        #log.info(f"Updated policyLoss for {self.name}")
        
        return

    # TODO, Update this
    def calculateAndUpdateL_C(self):
        """
        vTarget detached because it is assumed to be correct and thus should not be the variable that is
        effected by loss.
        On the other hand vPredicted should be alligned in a way to reduces loss, hence model to be modifed
        by backpropagation keepint vPredicted attached
        #"""
        pred = self.vPredicted
        targ = self.vTarget.detach()
        loss = torch.mean(torch.pow(pred-targ,2))
        log.info(f"Online: Critic loss = {loss}")
        self._updateCritc(loss)
        log.info(f"Online: Updated criticLoss for {self.name}")
        return

    def auxillaryForward(self,states):
        if self.name=='GOD':
            return self.forwardP(states)
        else: return self.god.forwardP(states)

    def saveModel(self,path):
        condition = path +"/"+str(self.__nAgent)+"_"+str(self.maxEpisode)+"_"+str(self.trajectoryLength)+"_"+str(len(self._actionSpace))+"_"+str(self.__actorLR)+"_"+str(self.__criticLR)+"_"
        self.__policyNet.saveM(condition+"PolicyModel.pt")
        self.__criticNet.saveM(condition+"CritcModel.pt")
        return

    def __loadModel(self,path):
        """
        If using GPU, this has to be mapped to it while load .. torch.load(path,map_location=device)
        #"""
        curr = multiprocessing.current_process()
        #self._policySemaphore.acquire()
        self.__policyNet.loadM(path+"PolicyModel.pt")
        #self._policySemaphore.release()
        #self._criticSemaphore.acquire()
        self.__criticNet.loadM(path+"CritcModel.pt")
        #self._criticSemaphore.release()
        return


class BOSS(GOD):
    """
   The actual class which does the exploration of the state space.
   Contains the code for the actor critic algorithm (Trajectory generation+ Policy gradient and value net updation )

   @Input :: (From Enviorenment) The current state + next state according to the current price to create trajectory.

   @Output:: (To Enviorenment)   The action taken for creating trajectory.

   @Actual Job :: To update the policy network and critic net by creating the trajectory and calculating losses.


    #"""
    def __init__(self,
                 maxEpisode,
                 god,
                 trajectoryLength,
                 stateSize,
                 name,
                 gamma=0.9,   # Decreaing, so that later rewards matter less
                 # lamda=0.1, # Lambda was earlier used for GAE
                 # depth=200, # Not used anymore
                 debug=False,
                 ):
        super(BOSS,self).__init__(maxEpisode=maxEpisode,debug=debug,trajectoryLength=trajectoryLength,stateSize=stateSize,name=name)
        self.god = god
        self._actionSpace = self.god._actionSpace
        log.info(f"{self.name}\tAction Space\t{self._actionSpace}")
        self.trajectoryS = torch.zeros([self.trajectoryLength,self.stateSize])
        self.trajectoryR = torch.zeros(self.trajectoryLength)
        self.trajectoryA = torch.zeros(self.trajectoryLength)
        self.ɤ = gamma
        self.gamma=gamma
        # self.d = depth # Not using anymore
        self.vPredicted = torch.zeros(self.trajectoryLength)
        self.vTarget = torch.zeros(self.trajectoryLength)
        self.advantage = torch.zeros(self.trajectoryLength)
        # If entropy H_t calculated, Init beta
        pass

    def train(
            self,
            #resetSemaphore, # Not using resetSemaphore anymore
    ):
        self.makeENV()
        log.info(f"{self.name}'s Env made")

        """
        The Actual function to train the network , the actor-critic actual logic.
        Eviorienment.step :: env.step has to give different outputs for different state trajectories by
        different boss. it has to take in account the diff trajectories becouse diff bosses will go to
        different states.
        #"""
        curr = multiprocessing.current_process()
        if self.debug:
            log.debug(f"{self.name},currentP name,id,pid {curr.name},{curr.ident},{curr.pid}")
        """
        Here the main logic of training of A2C will be present
        `with factory.create(int(self.name[-2:]),nAgent) as progress:`
        This was for a multiproccessing TQDM, which failed.
        """
        for e in tqdm(range(self.maxEpisode),ascii=True,desc=self.name):
            if self.debug:
                log.debug(f"{self.name} e = {e}")
            resetState = self.reset()
            self.startState = torch.Tensor(resetState)
            if self.debug:
                log.debug(f"{self.name} Start state = {self.startState}")
            self.gatherAndStore()
            log.info(f"rewards = {self.trajectoryR}")

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

            """
            Question to be figured out :: Exactly when should the boss agents update the networks??
            #"""
            self.calculateAndUpdateL_P()
            # calculate  policy loss and update policy network
            self.calculateAndUpdateL_C()
            # calculate critic loss and update critic network
            log.info(f"{self.name} episode {e} Completed")
            #progress.update(1)
        log.info(f"{self.name} has completed training")
        print(f"{self.name} has completed training")
        pass

    # For creating an instance of env for each worker (BOSS) agent to assist in training 
    def makeENV(self):
        """
        Making enviornment here
        Because of #35472 on pytorch : https://github.com/pytorch/pytorch/issues/35472#issue-588591481
        Proposed solution is: https://github.com/pytorch/pytorch/issues/35472#issuecomment-604775738
        #"""
        LSTM_instance = LSTM(output_size, input_dim, hidden_dim, layer_dim,debug=self.debug)
        if self.debug:
            log.info(f"LSTM instance created for {self.name} = {LSTM_instance}")


        LSTM_instance.loadM("ENV_MODEL/lstm_modelV3.pt")
        log.info(f"{self.name}'s Env made")

        log.info(f"LSTM instance loaded for {self.name} = {LSTM_instance}")
        self._env = ENV(
            model=LSTM_instance,
            dataset_path=envDATA,
            actionSpace=self._actionSpace,
            debug=self.debug,
        )
        self.reset()
        return

    # Gathers trajectories of states,actions,rewards accumulated over the course of time by acting on the environment using current policy
    def gatherAndStore(self):
        
        # Initial starting state 
        currentState = self.startState
        log.info(f"Starting state={currentState}, for {self.name}")
        for i in range(self.trajectoryLength):
            # Decide action to take from currentState
            action = self.getAction(currentState)
            #nextState,reward,info = self.god.step(action)
            # Take action in the env and observe nextState,reward and other relevant params
            nextState,reward,_,info = self.step(action)
            ## Oi generous env , please tell me the next state and reward for the action i have taken
            log.info(f"{self.name}, {i},  {info}")
            if self.debug:
                log.debug(f"Reward and Shape = {reward}, {reward.shape}")
            # Update and store trajectories 
            self.trajectoryS[i] = currentState
            self.trajectoryA[i] = action
            self.trajectoryR[i] = torch.Tensor(reward)
            if self.debug:
                log.debug(f"Action for {self.name} {i} = {action}, {type(action)}")
                log.debug(f"Detached Next state {nextState}")
            # Change currentState to nextState after action has been taken
            currentState=torch.Tensor(nextState)

        if self.debug:
            log.debug(f"Action = {self.trajectoryA}")
            log.debug(f"vPred = {self.vPredicted}")
            log.debug(f"vTar = {self.vTarget}")
        pass
    
    # To get action based on the state given, returns actionIndex based on actionSpace
    def getAction(self,state):
        actionIndex,probab = self.god.decideAction(state)
        return actionIndex

    # def calculateV_p(self):
    #     # calculate the predicted v value by using critic network
    #     # Predicted value is just the value returned by the critic network.
    #     self.vPredicted = Tensor(len(self.vPredicted))
    #     # This resets the tensor to zero
    #     for i in range(self.trajectoryLength):
    #         state=self.trajectoryS[i]
    #         self.vPredicted[i]=self.god._getCriticValue(state)
    #     return

    # def calculateV_tar(self):
    #     # calculate the target value v_tar using critic network
    #     """
    #     This is a huge topic of debate , i.e how to actually calculate the target value, currently we have 2 propositions.
    #     1. v_target(s) = summation( reward + v_predicted(ss)) , where ss is some state after the trajectory.
    #     2. calculate v_target with the help of advantage function itself.

    #     #####################################################################################################
    #     Another Huge Doubt regarding v_target and GAE is::

    #     DO WE NEED TO CONSTRUCT NEW EXPERIENCE(ENVIORENMENT EXPLORATION) FOR CALCULATING V_tar OR WE CAN USE THE
    #     CURRENT TRAJECTORY FOR THIS PURPOSE(more probable)

    #     CHOICE 1 :: Use TD error , for each state in trajectory , use TD error to calculate V_tar since
    #     you know 𝛄 + next state reward and you can calculate value of the next and current state
    #     Pros :: very fast and TD error is a reliable and solid method.
    #     Cons :: Maybe if we only see one step ahead , the estimate will be less reliable.

    #     CHOICE 2 :: For this we will, for each state in the trajectory, calculate the advantage and V_tar
    #     using the previous method (by travelling to the end of the trajectory and accumulating rewards as
    #     given in jamboard slide 15) , the only difference is we start from the current state itself to the
    #     end of trajectory. (Or until a depth)

    #     We have chosen choice 2 for v_tar , by terating in reverse direction in the trajectory list.
    #     #"""
    #     # we have set γ to be 0.99 // see this sweet γ @BlackD , α , β , θ 
    #     ## here 𝛄 can be variable so, the length can be changed.
    #     #  ans=0.0
    #     #  for i in range(0,200):
    #     #      ans+=((self.ɤ)**(i+1))*self.trajectory[i][2]
    #     # ans+=(self.ɤ)**200*self.god._getCriticValue((self.trajectory[200][0])) ## multiply by the actual value of the 200th state.
    #     #  return ans

    #     self.vTarget = Tensor(len(self.vTarget))
    #     self.vTarget[self.trajectoryLength-1] = self.trajectoryR[self.trajectoryLength-1]
    #     ## only the reward recieved in the last state , we can also put it zero i think
    #     # guess will have to consult literature on this, diff shouldn't be substantial.
    #     for i in reversed(range(self.trajectoryLength-1)):
    #         # iterate in reverse order.
    #         self.vTarget[i] = self.trajectoryR[i].clone() + self.ɤ*self.vTarget[i+1].clone()
    #         # v_tar_currentState = reward + gamma* v_tar_nextState
    #     return

    def calculateNSTEPAdvantage(self):
        """
        Calculate Advantage using TD error/N-STEP , logic similar to vTarget calculation
        #"""
        vPredLast = self.vPredicted[self.trajectoryLength-1]
        self.advantage =  Tensor(len(self.advantage))
        self.advantage[-1]=vPredLast
        for i in reversed(range(self.trajectoryLength-1)):
            self.advantage[i] = self.trajectoryR[i].clone() + self.ɤ*self.advantage[i+1].clone() - self.vPredicted[i].clone()
        log.info(f"Advantage {self.name} = {self.advantage}")
        return

    # def calculateAndUpdateL_P(self):
    #     ### Semaphore stuff for safe update of network by multiple bosses.
    #     """
    #     FOR UPDATING THE ACTOR USING POLICY GRADIENT WE MUST CALCULATE THE LOG PROBABILITY OF ACTION GIVEN
    #     A PARTICULAR STATE.BUT IN SITUATION OF MULTIPLE AGENTS IT MAY HAPPEN THAT BEFOR AGENT 1 FETCHES THIS
    #     PROBABILITY AND UPDATES AND UPDATES THE NETWORK , AGENT-2 MAY HAVE TAMPERED/UPDATED THE NETWORK.

    #     TO WORK AROUND THIS 2 CHOICES::
    #     CHOICE 1 :: KEEP TRACK OF THE STATE ACTION PROBABILITY FOR EACH AGENT'S TRAJECTORY , SO EVEN IF
    #     ANOTHER AGEND UPDATES THE NET, WE HAVE THE EXACT ORIGINAL PROBABILITY FROM WHEN THE AGENT HAD SAMPLED
    #     THE ACTION.

    #     CHOICE 2 :: BE IGNORANT , THE CHANCE THAT 2 AGENT HAVE TAMPERED WITH THE SAME STATE BEFOR UPDATING
    #     THE NETWORK WITH THEIR OWN LOSS IS EXTREMELEY LOW, SO IT DOESN'T MATTERS.

    #     # Advantage detached because: Advantage is the result of critc, and if it is backpropagated here,
    #     critic model might face issues during it's own backpropagation
    #     #"""

    #     # TODO, Update this
    #     # TODO check if this works, also if its needed in forwardP
    #     pd = self.god.forwardP(self.trajectoryS)
    #     dist = Categorical(pd)
    #     logProb = dist.log_prob(self.trajectoryA)
    #     advantage = self.advantage.detach()
    #     if self.debug:
    #         log.debug(f" advantage detached for {self.name}")
    #     loss = -1*torch.mean(advantage*logProb)
    #     log.info(f"Policy loss = {loss}")
    #     self.god._updatePolicy(loss)
    #     log.info(f"Updated policyLoss for {self.name}")
    #     return

    # def calculateAndUpdateL_C(self):
    #     """
    #     vTarget detached because it is assumed to be correct and thus should not be the variable that is
    #     effected by loss.
    #     On the other hand vPredicted should be alligned in a way to reduces loss, hence model to be modifed
    #     by backpropagation keepint vPredicted attached
    #     #"""
    #     pred = self.vPredicted
    #     targ = self.vTarget.detach()
    #     loss = torch.mean(torch.pow(pred-targ,2))
    #     log.info(f"Critic loss = {loss}")
    #     self.god._updateCritc(loss)
    #     log.info(f"Updated criticLoss for {self.name}")
    #     return
