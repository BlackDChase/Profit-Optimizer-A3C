# Configurations

## Training Parameters
### Hyper Parameters
- Number of Agents: 5
- Number of Episodes: 500
- Trajectory Length: 50
- Actions: 8
- Actor Learning Rate: 0.001
- Critic Learning Rate: 0.007

### Reward Parameters
```py
            maxAllowed = self.min_max_values["max"][price_index]
            minAllowed = self.min_max_values["min"][price_index]

            correction = 1
            if ((demand-supply) <0) or (new_price<minAllowed) or (new_price>correction):
                correction=0-abs(correction)
            if denormalize:
                correction/=(10**8)
            reward = (abs(demand - supply)**2) * abs(new_price) * correction
            log.info(f"State set = {new_price}, {correction}, {demand}, {supply}")
            return reward
```

## Model
### LSTM Env
```py
LSTM(
  (lstm): LSTM(13, 128, batch_first=True)
  (fc): Linear(in_features=128, out_features=13, bias=True)
  (norm): Hardtanh(min_val=-0.5, max_val=15)
)

```
### Policy Network
```py
Network(
  (hypoThesis): Sequential(
    (0): Linear(in_features=13, out_features=40, bias=True)
    (1): Hardtanh(min_val=-300, max_val=50000)
    (2): Linear(in_features=40, out_features=40, bias=True)
    (3): SELU()
    (4): Linear(in_features=40, out_features=50, bias=True)
    (5): Sigmoid()
    (6): Linear(in_features=50, out_features=8, bias=True)
    (7): Softmax(dim=0)
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
            L1=(nn.Linear,40,nn.SELU()),
            L2=(nn.Linear,40,nn.Sigmoid()),
            L3=(nn.Linear,50,nn.Softmax(dim=0)),
            debug=self.debug,
            ## we will add softmax at end , which will give the probability distribution.
        )
        ```
### Critic Network
#### Layout
```py
Network(
  (hypoThesis): Sequential(
    (0): Linear(in_features=13, out_features=30, bias=True)
    (1): Hardtanh(min_val=-300, max_val=50000)
    (2): Linear(in_features=30, out_features=40, bias=True)
    (3): SELU()
    (4): Linear(in_features=40, out_features=1, bias=True)
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
            L1=(nn.Linear,30,nn.SELU()),
            L2=(nn.Linear,40,nn.Tanh()),
            debug=self.debug,
        )
```
