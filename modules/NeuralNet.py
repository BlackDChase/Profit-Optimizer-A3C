import torch
import log
from torch import nn, device
# GLOBAL
#device = device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    #global device
    def __init__(self,stateSize,actionSize,name,debug=False,lr=1e-3,**kwargs):
        #print(kwargs)
        #sys.exit()
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
        self.debug = debug
        self.name = name
        layers = []
        keyWords = list(kwargs.keys())
        kwargs["stateSize"] = (nn.Linear,stateSize,nn.ReLU())

        """
        Input layer
        #"""
        keyWords.insert(0,"stateSize")
        i=0
        if self.debug:
            log.debug(f"kwargs for {self.name} = {kwargs}")
            log.debug(f"actionSize is {actionSize}")
            log.debug(f"stateSize is {stateSize}")
        for i in range(len(keyWords)-1):
            l1=keyWords[i]
            l2=keyWords[i+1]
            """
            kwargs[l1][0] : name of the layer
            kwargs[l1][1] : inputSize of the layer
            kwargs[l2][1] : outputSize pf the layer == inputSize of next layer
            kwargs[l1][2] : activation function of the layer
            #"""
            layers.append(kwargs[l1][0](in_features=kwargs[l1][1],out_features=kwargs[l2][1]))
            if len(kwargs[l1])>=3:
                """
                For layers with activation function
                #"""
                layers.append(kwargs[l1][2])
        l1=keyWords[len(keyWords)-1]
        layers.append(kwargs[l1][0](kwargs[l1][1],actionSize))
        """
        Output Layer
        #"""
        if len(kwargs[l1])==3:
            """
            For layers with activation function
            #"""
            layers.append(kwargs[l1][2])
        self.myLayers=layers
        self.hypoThesis = nn.Sequential(*layers)
        self.params = self.hypoThesis.parameters()

        if self.debug:
            log.debug(f"Layers for {self.name} = {layers}")
            log.debug(f"{self.name} Model: {self.hypoThesis}")

        """
        Initiallising model
        Rest parameters to uniform
        0 is lower bound, 1 is upper bound
        But thats not working
        """
        #nn.init.uniform_(self.params,0,1)
        """
        Optimizer and loss function
        #"""
        self.optimizer = torch.optim.SGD(self.params,lr=self.learningRate)
        #self.optimizer = torch.optim.Adam(self.params,lr=self.learningRate)
        pass

    def forward(self,currentState):
        """
        Override for forward from nn.Module
        To calculate the pd-parameters by using the neural net.
        __________
        Output for Policy network   : Probability Distribution Parameter
        Output for Critic network   : Advantage
        #"""
        output = self.hypoThesis(currentState)
        if self.debug:
            log.debug(f"current state for {self.name} : {currentState}")
            log.debug(f"output of model = {output}")
        return output
    pass

class Aktor(nn.Module):
    def __init__(self):
        super(Aktor,self).__init__()
        self.learningRate = 1e-3
        self.model = nn.Sequential(
            nn.Linear(in_features=9,out_features=20),
            nn.Tanh(),
            nn.Linear(in_features=20,out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50,out_features=11),
            nn.Softmax(),
        )
        self.params = self.model.parameters()
        self.optimizer = torch.optim.SGD(self.params,lr=self.learningRate)
    def forward(self,currentState):
        output = self.model(currentState)
        return output

class Kritc(nn.Module):
    def __init__(self):
        super(Kritc,self).__init__()
        self.learningRate = 1e-3
        self.model = nn.Sequential(
            nn.Linear(in_features=9,out_features=20),
            nn.ELU(),
            nn.Linear(in_features=20,out_features=1),
            nn.LeakyReLU(),
        )
        self.params = self.model.parameters()
        self.optimizer = torch.optim.SGD(self.params,lr=self.learningRate)
    def forward(self,currentState):
        output = self.model(currentState)
        return output
