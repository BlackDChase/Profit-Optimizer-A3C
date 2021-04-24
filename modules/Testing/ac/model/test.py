# Testing multi processing 
# Should be in module foder 
# this code may have spelling mistake 
import torch
from NeuralNet import Aktor, Kritc, Network
import multiprocessing
import log

debug=True

def run(model,state):
    log.info(f"{model.name} Adderess = {hex(id(model))}")
    log.info(f"State shape = {state.shape}")
    x=model(state)
    log.info(f"Result shape = {x.shape}")

    return True


multiprocessing.set_start_method('fork')
threads = []
output_size = 13
input_dim = output_size
hidden_dim = 128
layer_dim = 1
policyNet = Network(
    13,
    7,
    lr=0.01,
    name="Policy Net",
    L1=(nn.Linear,20,nn.Tanh()),
    L2=(nn.Linear,50,nn.Softmax(dim=0)),
    debug=debug,
    ## we will add softmax at end , which will give the probability distribution.
)

# Could be updated
## the critic network :: it's input is state and output is a scaler ( the value of the state)

criticNet = Network(
    13,
    1,
    lr=0.01,
    name="Critic Net",
    L1=(nn.Linear,30,nn.ReLU6()),
    L2=(nn.Linear,40,nn.ReLU6()),
    debug=self.debug,
)
aktorNet = Aktor()
kriticNet = Kritc()

states = [torch.rand(10,13),torch.rand(13)]
models = [policyNet,criticNet,aktorNet,kriticNet]



for model in models:
    print("To show that loaded model worked without multiprocessing")
    for j in states:
        state=j
        run(model,state)

    print("Worked I {model.name}, states show results with loaded model and no multiprocessing")




    # This line is not actually needed
    model.share_memory()

    print("To show that model works with multiprocessing")
    for j in states:
        state=j
        threads=[]
        for i in range(5):
            p = multiprocessing.Process(target=run,args=(model,state,))
            threads.append(p)

        for i in threads:
            i.start()
        for i in threads:
            i.join()
        print(f"Worked II {j} {model.name}, Saved model worked with multiprocessing")

    print(f"Worked III {model.name}, Saved model worked with multiprocessing for all states")
