# Configurations

## Training Parameters
### Hyper Parameters
- Number of Agents: 20
- Number of Episodes: 700
- Trajectory Length: 70
- Actions: 8
- Actor Learning Rate: 0.002
- Critic Learning Rate: 0.009
- Average Profit : 
### Reward Parameters
```py
        maxAllowed = self.min_max_values["max"][price_index]
        minAllowed = self.min_max_values["min"][price_index]

        correction = 1
        if ((demand-supply) <0) or (new_price<minAllowed) or (new_price>maxAllowed):
            correction=0-abs(correction)

        if denormalize:
            correction/=(10**15)
        reward = (abs(demand - supply)**3) * (abs(new_price)**2) * correction
        log.info(f"State set = {new_price}, {correction}, {demand}, {supply}")
        return reward
```

## Model
### LSTM Env
```py
LSTM(
  (lstm): LSTM(13, 40, num_layers=2, batch_first=True)
  (fc): Linear(in_features=40, out_features=13, bias=True)
  (norm): Hardtanh(min_val=-0.01, max_val=2)
)
```
### Policy Network
```py
Network(
  (hypoThesis): Sequential(
    (0): Linear(in_features=13, out_features=18, bias=True)
    (1): Hardtanh(min_val=-3, max_val=6)
    (2): Linear(in_features=18, out_features=18, bias=True)
    (3): SELU()
    (4): Linear(in_features=18, out_features=8, bias=True)
    (5): Softmax(dim=0)
  )
)
```
#### Code
```py
        self.__policyNet = Network(
            self.stateSize,
            len(self._actionSpace),
            lr=self.__actorLR,
            name="Policy Net",
            L1=(nn.Linear,18,nn.SELU()),
            L2=(nn.Linear,18,nn.Softmax(dim=0)),
            debug=self.debug,
            ## we will add softmax at end , which will give the probability distribution.
        )
        ```
### Critic Network
#### Layout
```py
Network: Network(
  (hypoThesis): Sequential(
    (0): Linear(in_features=13, out_features=10, bias=True)
    (1): Hardtanh(min_val=-3, max_val=6)
    (2): Linear(in_features=10, out_features=10, bias=True)
    (3): SELU()
    (4): Linear(in_features=10, out_features=1, bias=True)
    (5): Tanh()
  )
)
```
#### Code
```py
        self.__criticNet = Network(
            self.stateSize,
            1,
            lr=self.__criticLR,
            name="Critic Net",
            L1=(nn.Linear,10,nn.SELU()),
            L2=(nn.Linear,10,nn.Tanh()),
            debug=self.debug,
        )
```
