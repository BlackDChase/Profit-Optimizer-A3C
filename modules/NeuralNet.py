"""
Contains Various Neural Nets that an Agent can use
Primarily We would be using Network Class as it is more robust to change
Aktor, Kritc are standins for testing
#"""
__author__ = 'BlackDChase,MR-TLL'
__version__ = '1.2.6'

# Imports
import torch
import log
from torch import nn
import multiprocessing
#GLOBAL
#device = device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    #global device
    def __init__(self,stateSize,outputSize,name,debug=False,lr=1e-3,**kwargs):
        #print(kwargs)
        #sys.exit()
        """
        ## All logic below are initializing the Network with the layers.
        Parameters:
        __________
        : stateSize     : Input Size
        : outputSize    : Output Size
        : **kwargs      : Keyword arguments for different layers should be L1,L2,L3..
                          (because its a dic and will change the oredring atuomatically)
                          Later we kan have it Ordered dic to be directly sent throught
                          but would make less sense
                          The touple has type of layer, it's input and activation funtion if it's there.
        #"""
        super(Network,self).__init__()
        self.inputSize = stateSize
        self.outputSize = outputSize
        self.learningRate = lr
        self.debug = debug
        self.name = name
        layers = []
        keyWords = list(kwargs.keys())
        kwargs["stateSize"] = (nn.Linear,stateSize,nn.Hardtanh(min_val=-3,max_val=6))

        """
        Input layer
        #"""
        keyWords.insert(0,"stateSize")
        i=0
        if self.debug:
            log.debug(f"kwargs for {self.name} = {kwargs}")
            log.debug(f"outputSize is {outputSize}")
            log.debug(f"stateSize is {stateSize}")
        for i in range(len(keyWords)-1):
            l1=keyWords[i]
            l2=keyWords[i+1]
            """
            kwargs[l1][0] : name of the layer
            kwargs[l1][1] : inputSize of the layer
            kwargs[l1][2] : activation function of the layer
            kwargs[l2][1] : outputSize pf the layer == inputSize of next layer
            #"""
            layers.append(kwargs[l1][0](in_features=kwargs[l1][1],out_features=kwargs[l2][1]))
            if len(kwargs[l1])>=3:
                """
                For layers with activation function
                #"""
                layers.append(kwargs[l1][2])
        l1=keyWords[len(keyWords)-1]
        layers.append(kwargs[l1][0](kwargs[l1][1],outputSize))
        """
        Output Layer
        #"""
        activation=2
        while len(kwargs[l1])>activation:
            """
            For layers with one or more activation function
            #"""
            layers.append(kwargs[l1][activation])
            activation+=1
        self.myLayers=layers
        self.hypoThesis = nn.Sequential(*layers)
        self.params = self.hypoThesis.parameters()

        if self.debug:
            log.debug(f"Layers for {self.name} = {layers}")
            log.debug(f"{self.name} Model: {self.hypoThesis}")

        """
        Initializing model
        Rest parameters to uniform
        0 is lower bound, 1 is upper bound
        But thats not working
        """
        #nn.init.uniform_(self.params,0,1)
        """
        Optimizer and loss function
        #"""
        #self.optimizer = torch.optim.SGD(self.params,lr=self.learningRate)
        self.optimizer = torch.optim.Adam(self.params,lr=self.learningRate)
        pass

    def forward(self,currentState):
        """
        Override for forward from nn.Module
        To calculate the pd-parameters by using the neural net.
        __________
        Output for Policy network   : Probability Distribution Parameter
        Output for Critic network   : Advantage
        #"""
        output=[]
        curr = multiprocessing.current_process()
        if self.debug:
            log.debug(f"current state for {self.name} of {curr.ident} : {currentState}")

        if len(currentState.shape)>1:
            """
            With multi threading, mulitple states are not parsed through the
            model, they get stuck.
            We therefore parse them one state at a time, then make the stack
            and return it.
            """
            if self.debug:
                log.debug(f"Shape of current state {self.name} of {curr.ident} = {currentState.shape}")
            for i in currentState:
                step = self.hypoThesis(i)
                output.append(step)
            output = torch.stack(output)
        else:
            output = self.hypoThesis(currentState)
        if self.debug:
            log.debug(f"output of model for {self.name} of {curr.ident} = {output}")
        return output

    def saveM(self,name):
        curr = multiprocessing.current_process()
        torch.save(self.hypoThesis.state_dict(),name)
        log.info(f"{self.name} saved = {self.hypoThesis}")
        if self.debug:
            log.debug(f"Model saved for {self.name} of {curr.ident} : {self.hypoThesis}")

    def loadM(self,path):
        curr = multiprocessing.current_process()
        self.hypoThesis.load_state_dict(torch.load(path))
        log.info(f"{self.name} loaded = {self.hypoThesis}")
        if self.debug:
            log.debug(f"Model loaded for {self.name} of {curr.ident} : {self.hypoThesis}")
    pass

# This was the proposed predefined testing model class for handling and managing actor network 
# which is now replaced with the more general Network class to avoid conflicts
class Aktor(nn.Module):
    def __init__(self):
        super(Aktor,self).__init__()

        # Default learning rate
        self.learningRate = 1e-3
        # Actor Network internal sequential layers 
        self.model = nn.Sequential(
            nn.Linear(in_features=13,out_features=20),
            nn.Linear(in_features=20,out_features=11),
            nn.Softmax(),
        )
        self.params = self.model.parameters()
        self.optimizer = torch.optim.SGD(self.params,lr=self.learningRate)

    def forward(self,currentState):
        output = self.model(currentState)
        return output

    def saveM(self,name):
        torch.save(self.model.state_dict(),name)
        log.info(f"Kritc saved = {self.model}")

    def loadM(self,path):
        print("Trying to load")
        self.model.load_state_dict(torch.load(path))
        log.info(f"Kritc loaded = {self.model}")
    pass

# This was the proposed predefined testing model class for handling and managing critic network 
# which is now replaced with the more general Network class to avoid conflicts 
class Kritc(nn.Module):
    def __init__(self):
        super(Kritc,self).__init__()

        # Default learning rate
        self.learningRate = 1e-3
        # Critic Network internal sequential layers 
        self.model = nn.Sequential(
            nn.Linear(in_features=13,out_features=20),
            nn.ELU(),
            nn.Linear(in_features=20,out_features=1),
            nn.LeakyReLU(),
        )
        self.params = self.model.parameters()
        self.optimizer = torch.optim.SGD(self.params,lr=self.learningRate)

    def forward(self,currentState):
        output = self.model(currentState)
        return output

    def saveM(self,name):
        torch.save(self.model.state_dict(),name)
        log.info(f"Kritc saved = {self.model}")

    def loadM(self,path):
        print("Trying to load")
        self.model.load_state_dict(torch.load(path))
        log.info(f"Kritc loaded = {self.model}")
    pass
