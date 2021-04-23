# Testing multi processing 
# Should be in module foder 
# this code may have spelling mistake 
import torch
from lstm import LSTM
import multiprocessing
import log

debug=True

def run(model,state):
    log.info(f"Model Adderess = {hex(id(model))}")
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
model = LSTM(output_size, input_dim, hidden_dim, layer_dim,debug=debug)
states=[torch.rand(1,10,13),torch.rand(10,10,13),torch.rand(1,1,13),torch.rand(10,1,13)]



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
print("Worked I, all threads joined")



# It works if model is not loaded (next one line commented)
model.loadM('ENV_MODEL/lstm_model.pt')

print("To show that loaded model worked without multiprocessing")
for j in states:
    state=j
    run(model,state)

print("Worked II, states show results with loaded model and no multiprocessing")




# This line is not actually needed
model.share_memory()

print("To show that loaded model worked dosnt work with multiprocessing")
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
    print(f"Worked III {j}, Saved model worked with multiprocessing")

print(f"Worked IV, Saved model worked with multiprocessing for all states")
