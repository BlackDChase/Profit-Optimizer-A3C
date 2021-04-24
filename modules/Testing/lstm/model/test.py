# Testing multi processing 
# Should be in module foder 
# this code may have spelling mistake 
import torch
from lstm import LSTM
from torch import multiprocessing,nn
import log

debug=True
multiprocessing.set_start_method('fork')
output_size = 13
input_dim = output_size
hidden_dim = 128
layer_dim = 1

outModel = LSTM(output_size, input_dim, hidden_dim, layer_dim,debug=debug)
outModel.saveM("ENV_MODEL/trial")

## LSTM
#lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
#torch.save(lstm.state_dict(),"ENV_MODEL/plain.pt")


def run(state):
    model = LSTM(output_size, input_dim, hidden_dim, layer_dim,debug=debug)
    model.loadM('ENV_MODEL/lstm_model.pt')
    #model.loadM('ENV_MODEL/trial.pt')
    #model = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
    #model.load_state_dict(torch.load("ENV_MODEL/plain.pt"))
    log.info(f"Model Adderess = {hex(id(model))}")
    log.info(f"State shape = {state.shape}")
    x=model(state)
    #x=torch.Tensor(x)
    log.info(f"Result shape = {x.shape}")
    print(x.shape)
    return True



states=[torch.rand(1,10,13),torch.rand(10,10,13),torch.rand(1,1,13),torch.rand(10,1,13)]


"""
print("To show that model works with multiprocessing")
for j in states:
    state=j
    threads=[]
    for i in range(5):
        p = multiprocessing.Process(target=run,args=(state,))
        threads.append(p)

    for i in threads:
        i.start()
    for i in threads:
        i.join()
print("Worked I, all threads joined")



# It works if model is not loaded (next one line commented)

print("To show that loaded model worked without multiprocessing")
for j in states:
    state=j
    run(state)

print("Worked II, states show results with loaded model and no multiprocessing")

#"""


# This line is not actually needed
#model.share_memory()

print("To show that loaded model worked dosnt work with multiprocessing")
for j in states:
    try:
        state=j
        threads=[]
        # Works
        #for i in range(500):
        for i in range(5):
            p = multiprocessing.Process(target=run,args=(state,))
            threads.append(p)

        for i in threads:
            i.start()
        for i in threads:
            i.join()
        print(f"Worked III {j.shape}, Saved model worked with multiprocessing")
    except KeyboardInterrupt:
        print(f"Failed III {j.shape}")
        pass

print(f"Worked IV, Saved model worked with multiprocessing for all states")
