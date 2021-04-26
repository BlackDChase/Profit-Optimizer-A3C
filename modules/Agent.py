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

State = Ontario Price, Market Demand,Ontario Demand,Northwest,Northeast,Ottawa,East,Toronto,Essa,Bruce, Northwest Nigiria, West
"""
__author__ = 'BlackDChase,MR-TLL'
__version__ = '0.3.0'

# Imports
from torch import nn, Tensor
from torch.distributions import Categorical
import torch
import numpy as np
from tqdm import tqdm
import log
import sys
from NeuralNet import Network #, Aktor, Kritc
import multiprocessing
from multiprocessing import Process, Lock
multiprocessing.set_start_method('fork')
"""
from TempEnv import TempEnv as ENV
"""
from env import LSTMEnv as ENV
from lstm import LSTM
output_size = 13
input_dim = output_size
hidden_dim = 128
layer_dim = 1
envDATA="../Dataset/col13data1.csv"
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
        self.stateSize = stateSize

        # state is the 9 dimentional tensor , defined at the top
        self._state = Tensor([0]*stateSize)

        # action space is the percent change of the current price.
        self._actionSpace = Tensor(actionSpace)
        self.debug = debug
        self.price = 0
        self.__nAgent=nAgent
        self.__bossAgent=[]
        if path==None:
            self.needTraining(maxEpisode,nAgent,trajectoryLength,alr,clr)
        else:
            self.loadModel(path)
        pass

    def needTraining(self,maxEpisode,nAgent,trajectoryLength,alr,clr):
        log.info("This GOD will train with the enviornment")
        self.setMaxEpisode(maxEpisode)
        self.setNumberOfAgent(nAgent)
        self.setTrajectoryLength(trajectoryLength)
        self.__actorLR = alr
        self.__criticLR = clr

        # semaphores do deal with simultaneous updates of the policy and critic net by multiple bosses.
        self._policySemaphore = Lock()
        self._criticSemaphore = Lock()
        self._resetSemaphore = Lock()
        #"""
        ## defining the actual neural net itself, input is state output is probability for each action.
        self.__policyNet = Network(
            self.stateSize,
            len(self._actionSpace),
            lr=self.__actorLR,
            name="Policy Net",
            L1=(nn.Linear,20,nn.Tanh()),
            L2=(nn.Linear,50,nn.Softmax(dim=0)),
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
            L1=(nn.Linear,30,nn.ReLU6()),
            L2=(nn.Linear,40,nn.ReLU6()),
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
        return

    def giveEnvironment(self,env):
        self.__env=env
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

    def train(self):
        self.__initateBoss()
        self.__trainBoss()
        return

    def test(self,time=100):
        currentState = self.__env.reset()
        a3cState=[]
        for i in range(time):
            log.info(f"a3cState={currentState}")
            a3cState.append(currentState)
            action = self._getAction(currentState)
            nextState,reward,_,info = self.__env.step(action)
            ## Oi generous env , please tell me the next state and reward for the action i have taken
            log.info(f"{self.name}, {i},  {info}")
            if self.debug:
                log.debug(f"Reward and Shape = {reward}, {reward.shape}")
                log.debug(f"Action for {self.name} {i} = {action}, {type(action)}")
            currentState=torch.Tensor(nextState)
        a3cState.append(currentState)
        a3cState = np.array(a3cState)
        return a3cState

    def compare(self,a3cState,time=100,normalState=None):
        if normalState==None:
            normalState=self.getNormalStates(time)
        profit=[]
        profitA3C=[]
        for i in range(len(a3cState)):
            profit.append(normalState[i][0]*(normalState[i][2]+normalState[i][1]))
            profitA3C.append(a3cState[i][0]*(a3cState[i][2]+a3cState[i][1]))
        return profit,profitA3C

    def getNormalStates(self,time=100):
        normalState=self.__env.possibleState(time)
        return normalState


    def _updatePolicy(self,lossP):
        curr = multiprocessing.current_process()
        if self.debug:
            log.debug(f"{curr.name},{curr.pid} Wants Policy")
        #self._policySemaphore.acquire()
        if self.debug:
            log.debug(f"Policy Semaphore Acquired, {curr.ident}")
        self.__policyNet.optimizer.zero_grad()
        lossP.backward(retain_graph=True)
        self.__policyNet.optimizer.step()
        #self._policySemaphore.release()
        if self.debug:
            log.debug(f"Policy Semaphore released, {curr.ident}")
        return

    def _updateCritc(self,lossC):
        curr = multiprocessing.current_process()
        if self.debug:
            log.debug(f"{curr.name},{curr.pid} Wants Critic")
        #self._criticSemaphore.acquire()
        if self.debug:
            log.debug(f"Critic Semaphore Acquired, {curr.ident}")
        self.__criticNet.optimizer.zero_grad()
        lossC.backward(retain_graph=True)
        self.__criticNet.optimizer.step()
        #self._criticSemaphore.release()
        if self.debug:
            log.debug(f"Critic Semaphore released, {curr.ident}")
        return
    
    '''
    def takeAction(self,state):
        """
        Take the final action according to the Policy Network.
        This method is not called inside GOD.
        This method is only called by ENV, every time it decides to take price for next time step.
        Enviorenment will send current state and this take action will return an action via env.step
        This will be done using pretrained policyNetwork.
        #"""
        actionProb = self._getAction(state)
        pd = Categorical(probs=actionProb)
        ## create a catagorical distribution acording to the actionProb
        ## categorical probability distribution
        actionIndex = pd.sample()
        probab = actionProb[actionIndex]
        nextState,reward,info = self.__env.step(actionIndex)
        nextState = torch.Tensor(nextState)
        if self.debug:
            log.debug(f"{self.name} step taken,{info}, rewards = {reward}")
            log.debug(f"Expected next State: {nextState}")
        return actionIndex,probab

    def _peakAction(self,state,action):
        """
        Online dilemma
        will be used at training time , for updating the networks
        #"""
        result = self.__env.step(state,action)
        return result
    #'''

    def _getAction(self,state):
        curr = multiprocessing.current_process()
        if self.debug:
            log.debug(f"{curr.name},{curr.pid} Wants Policy")
        #self._policySemaphore.acquire()
        if self.debug:
            log.debug(f"Policy Semaphore Acquired, {curr.ident}")
        actionProbab = self.__policyNet.forward(state)
        if self.debug:
            log.debug(f"Policy result = {actionProbab}")
            # Not sure if forward is the way to go
        #self._policySemaphore.release()
        if self.debug:
            log.debug(f"Policy Semaphore Released, {curr.ident}")
        return actionProbab

    def _getCriticValue(self,state):
        curr = multiprocessing.current_process()
        if self.debug:
            log.debug(f"{curr.name},{curr.pid} Wants Critic")
        #self._criticSemaphore.acquire()
        if self.debug:
            log.debug(f"Critic Semaphore Acquired, {curr.ident}")
        vVlaue = self.__criticNet.forward(state)
        #self._criticSemaphore.release()
        if self.debug:
            log.debug(f"Critic Semaphore Released, {curr.ident}")
        return vVlaue

    def reset(self):
        return torch.Tensor(self.__env.reset())

    def step(self,action):
        return self.__env.step(action)

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
            process = Process(target=self.__bossAgent[i].train,args=(self._resetSemaphore,))
            bossThreads.append(process)

        for i in range(len(bossThreads)):
            bossThreads[i].start()
            log.info(f"Boss{str(i).zfill(2)} training started via GOD")
        for i in bossThreads:
            i.join()

        """
        # Remove multiprocessing for @biribiri
        self.__bossAgent[0].train()
        if self.debug:
            log.debug(f"Boss00 training started via GOD")
        #"""
        return

    def forwardP(self,actions):
        curr = multiprocessing.current_process()
        if self.debug:
            log.debug(f"{curr.name},{curr.pid} Wants Policy")
        #self._policySemaphore.acquire()
        if self.debug:
            log.debug(f"Policy Semaphore Acquired, {curr.ident}")
        if self.debug:
            log.debug(f"{curr.ident} Actions {actions.shape}")
        actionProb = self.__policyNet.forward(actions)
        #self._policySemaphore.release()
        if self.debug:
            log.debug(f"Policy Semaphore Released, {curr.ident}")

        return actionProb

    def saveModel(self,path):
        condition= path +"/"+str(self.__nAgent)+"_"+str(self.maxEpisode)+"_"+str(self.trajectoryLength)+"_"+str(len(self._actionSpace))+"_"+str(self.__actorLR)+"_"+str(self.__criticLR)+"_"
        self.__policyNet.saveM(condition+"PolicyModel.pt")
        self.__criticNet.saveM(condition+"CritcModel.pt")
        return

    def loadModel(self,path):
        """
        If using GPU, this has to be mapped to it while load .. torch.load(path,map_location=device)
        #"""
        curr = multiprocessing.current_process()
        if self.debug:
            log.debug(f"{curr.name},{curr.pid} Wants Policy")
        #self._policySemaphore.acquire()
        if self.debug:
            log.debug(f"Policy Semaphore Acquired, {curr.ident}")
        self.__policyNet.loadM(path+"/PolicyModel.pt")
        #self._policySemaphore.release()
        if self.debug:
            log.debug(f"Policy Semaphore Released, {curr.ident}")

        curr = multiprocessing.current_process()
        if self.debug:
            log.debug(f"{curr.name},{curr.pid} Wants Critic")
        #self._criticSemaphore.acquire()
        if self.debug:
            log.debug(f"Critic Semaphore Acquired, {curr.ident}")
        self.__criticNet.loadM(path+"/CritcModel.pt")
        self._criticSemaphore.release()
        if self.debug:
            log.debug(f"Critic Semaphore Released, {curr.ident}")
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
                 gamma=0.99,
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
        # self.d = depth # Not using anymore
        self.vPredicted = torch.zeros(self.trajectoryLength)
        self.vTarget = torch.zeros(self.trajectoryLength)
        self.advantage = torch.zeros(self.trajectoryLength)
        # If entropy H_t calculated, Init beta
        pass
        #"""

    #def train(self,factory,nAgent):
    def train(self,resetSemaphore):
        self.makeENV(resetSemaphore)
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
            resetState = self.env.reset()
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

    def makeENV(self,resetSemaphore):
        """
        Making enviornment here
        Because of #35472 on pytorch : https://github.com/pytorch/pytorch/issues/35472#issue-588591481
        Proposed solution is: https://github.com/pytorch/pytorch/issues/35472#issuecomment-604775738
        #"""
        LSTM_instance = LSTM(output_size, input_dim, hidden_dim, layer_dim,debug=self.debug)
        if self.debug:
            log.info(f"LSTM instance created for {self.name} = {LSTM_instance}")

        #resetSemaphore.acquire()
        log.info(f"{resetSemaphore} acquired by {self}")

        LSTM_instance.loadM("ENV_MODEL/lstm_model.pt")
        log.info(f"{self.name}'s Env made")

        log.info(f"LSTM instance loaded for {self.name} = {LSTM_instance}")
        self.env = ENV(model=LSTM_instance,dataset_path=envDATA,actionSpace=self._actionSpace,debug=self.debug)
        self.env.reset()

        #resetSemaphore.release()
        log.info(f"{resetSemaphore} released by {self}")

        return

    def gatherAndStore(self):
        # gather a trajectory by acting in the enviornment using current policy
        currentState = self.startState
        log.info(f"Starting state={currentState}, for {self.name}")
        for i in range(self.trajectoryLength):
            action = self.getAction(currentState)
            #nextState,reward,info = self.god.step(action)
            nextState,reward,_,info = self.env.step(action)
            ## Oi generous env , please tell me the next state and reward for the action i have taken
            log.info(f"{self.name}, {i},  {info}")
            if self.debug:
                log.debug(f"Reward and Shape = {reward}, {reward.shape}")
            self.trajectoryS[i] = currentState
            self.trajectoryA[i] = action
            self.trajectoryR[i] = torch.Tensor(reward)
            if self.debug:
                log.debug(f"Action for {self.name} {i} = {action}, {type(action)}")
                log.debug(f"Detached Next state {nextState}")
            currentState=torch.Tensor(nextState)
        if self.debug:
            log.debug(f"Action = {self.trajectoryA}")
            log.debug(f"vPred = {self.vPredicted}")
            log.debug(f"vTar = {self.vTarget}")
        pass

    def getAction(self,state):
        """
        Responsible for taking the correct action from the given state using the neural net.
        @input :: current state
        @output :: the index of action which must be taken from this states, and probability of that action
        #"""
        state = state.float()
        actionProb = self.god._getAction(state)
        ## This creates state-action probability vector from the policy net. 
        pd = Categorical(probs=actionProb) ## create a catagorical distribution acording to the actionProb
        ## categorical probability distribution
        actionIndex = pd.sample() ## sample the action according to the probability distribution.

        return actionIndex

    def calculateV_p(self):
        # calculate the predicted v value by using critic network :: Predicted value is just the value returned by the critic network.
        self.vPredicted = Tensor(len(self.vPredicted))
        # This resets the tensor to zero
        for i in range(self.trajectoryLength):
            state=self.trajectoryS[i]
            self.vPredicted[i]=self.god._getCriticValue(state)
        return

    def calculateV_tar(self):
        # calculate the target value v_tar using critic network
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

        We have chosen choice 2 for v_tar , by terating in reverse direction in the trajectory list.
        #"""
        # we have set γ to be 0.99 // see this sweet γ @BlackD , α , β , θ 
        ## here 𝛄 can be variable so, the length can be changed.
        #  ans=0.0
        #  for i in range(0,200):
        #      ans+=((self.ɤ)**(i+1))*self.trajectory[i][2]
        # ans+=(self.ɤ)**200*self.god._getCriticValue((self.trajectory[200][0])) ## multiply by the actual value of the 200th state.
        #  return ans

        self.vTarget = Tensor(len(self.vTarget))
        self.vTarget[self.trajectoryLength-1] = self.trajectoryR[self.trajectoryLength-1]
        ## only the reward recieved in the last state , we can also put it zero i think
        # guess will have to consult literature on this, diff shouldn't be substantial.
        for i in reversed(range(self.trajectoryLength-1)):
            # iterate in reverse order.
            self.vTarget[i] = self.trajectoryR[i].clone() + self.ɤ*self.vTarget[i+1].clone()
            # v_tar_currentState = reward + gamma* v_tar_nextState
        return

    def calculateGAE(self):
        # calculate the Advantage using the critic network
        # gae is put on hold at this time
        # To be declared at latter stage
        raise NotImplementedError("You disturb BOSS for this?")
        pass


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

    def calculateAndUpdateL_P(self):
        ### Semaphore stuff for safe update of network by multiple bosses.
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

        pd = self.god.forwardP(self.trajectoryS)
        dist = Categorical(pd)
        logProb = dist.log_prob(self.trajectoryA)
        advantage = self.advantage.detach()
        if self.debug:
            log.debug(f" advantage detached for {self.name}")
        loss = -1*torch.mean(advantage*logProb)
        log.info(f"Policy loss = {loss}")
        self.god._updatePolicy(loss)
        log.info(f"Updated policyLoss for {self.name}")
        return

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
        log.info(f"Critic loss = {loss}")
        self.god._updateCritc(loss)
        log.info(f"Updated criticLoss for {self.name}")
        return
