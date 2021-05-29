# Adjusting Price to maximize profit using A3C 
`Version : '1.5.1'`

## How to Use
### Installing
It will setup a virtual environment and install all the required dependancies.
```
chmod +x install
./install
```
### Running (Train/Test)
#### Training
```
source bin/activate
./train.sh 
```
#### Testing
```
source bin/activate
./train.sh 
```
#### Advanced Training and Testing
1. If should shutdown the system once done `yes/N`.
2. If we should do a headstart by using a previusly trained model and fine tune it `True/false`.
3. Debugging, if logs shoul contain debugging or not `True/false`.
4. Testing methodology (all using a headstart and fine tuning over it).
    - `0`. No testing what so ever
    - `1`. Offline mode
    - `2`. Sliding Window Protocol
    - `3`. Episodic Protocol
    - `4`. All the protocols
5. Number of Indifferent Trains to perform. (1 is ideal, can opt 1-3 is recommended, keep the better perfomning one , delte the other)
6. Intrupting Test, as the model is for a live Electric Grid hence will not stop.
    - `n`. manually stopping by `KeyboardIntrupt`
    - `8`. Seconds after which it will Intrupt on its own.
```sh
#Points    (1) (2) (3) (4) (5) (6)
./run.sh   "y" "F" "T" "3" "2" "8"
```

## Ref
- [Research Paper](https://github.com/BlackDChase/Profit-Optimizer-A3C/blob/main/MARKDOWN/Reinforcement%20Learning%20for%20Optimizing%20the%20Demand%20Response.pdf)
- Authors:
    - [Harsh Kochar](https://github.com/BlackDChase/)
    - [Aadarsh Abhijat](https://github.com/MR-TLL/)
    - [Mrigyen Sawant](https://github.com/europe-asia-america/)
    - [Rithik Seth](https://github.com/ronin117-prog)
    - [Priyatam Somagattu](https://github.com/Pri1437)

## Metrics and Graphs
### A3C
#### Policy Gradient

[//]: # (Add Graph here)

[![Policy Gradient](https://github.com/BlackDChase/Profit-Optimizer-A3C/blob/main/MARKDOWN/)](https://github.com/BlackDChase/Profit-Optimizer-A3C/blob/main/MARKDOWN/)

#### Critic

[//]: # (Add Graph here)

[![Critic Loss](https://github.com/BlackDChase/Profit-Optimizer-A3C/blob/main/MARKDOWN/)](https://github.com/BlackDChase/Profit-Optimizer-A3C/blob/main/MARKDOWN/)

#### Advantage

[//]: # (Add Graph here)

[![Average Advantage](https://github.com/BlackDChase/Profit-Optimizer-A3C/blob/main/MARKDOWN/)](https://github.com/BlackDChase/Profit-Optimizer-A3C/blob/main/MARKDOWN/)

#### Reward

[//]: # (Add Graph here)

[![Average Reward](https://github.com/BlackDChase/Profit-Optimizer-A3C/blob/main/MARKDOWN/)](https://github.com/BlackDChase/Profit-Optimizer-A3C/blob/main/MARKDOWN/)

### REINFORCE
A previosly trained model For comparison. It uses [REINFORCE](https://github.com/BlackDChase/Profit-Optimizer-REINFORCE) a policy based algorithms.
#### Loss

[//]: # (Add Graph here)

#### Reward

[//]: # (Add Graph here)

#### Profit

[//]: # (Add Graph here)

### Environment
Both the algorithms were trained on an Environment which is based on an [LSTM](https://github.com/BlackDChase/Environment-Profit-Optimizer)
#### Loss

[//]: # (Add Graph here)


## Decisions
- Model has to be scaled in way that it can adjust for the following scenario
- Need to decide whether to use `reducedDF` or `advancedDF`
- Which Demand parameter to be used `Market Demand` or `Ontorio Demand`
- `Profit parameter` Should be of a time frame that is to be maximized
    - Can be `Demand*Price`
    - If it has to optimise profit for a longer term than t should be predicting for a longer term.
- __Environment is going to produce a reward__, The only thing matters for the agent is the reward , so it doesn't matters to agent how that reward is distributed, it will only chase the state which gives high reward. Now a preliminary reward can be (price * demand) , this needs to be refined later.


## PROFIT
- Bare profit of dataset 
    - Mean = 106272.98738089424
    - STD = 74348.89390813262
    - Min = 0.19366888086732809
    - Max = 5860463.267796092
    - Median = 104574.45803202249
- Profit of A3C model:

[//]: # (Add Graph here)

[![A3C Profit](https://github.com/BlackDChase/Profit-Optimizer-A3C/blob/main/MARKDOWN/)](https://github.com/BlackDChase/Profit-Optimizer-A3C/blob/main/MARKDOWN/)


## Bird Eye View
- God Agent:
    - Will take final action
    - A supervising psuedo agent that takes action
    - Methods
        - `__init__`
        - Get Current State
        - Take Action
        - Initiate Boss agents
        - Train Boss agents
    - Attributes
        - %Policy Network
        - %Critic Network
        - Max Episode
        - Trajectory Lenght 
        - Number of boss agents
        - List of Boss agents
        - State
        - Price
- Boss Agent(God Agent):
    - Will Update the network
    - Methods
        - `__init__`
        - train
        - Gather and Store `$(s_t,a_t,r_t,s_t\prime)$`
        - Calculate `v_p`
        - Calculate `v_t`
        - Calculate `gae`
        - Calculate and Update `l_p %`
        - Calculate and Update `l_c %`
    - Attributes
        - `$\alpha_A$`
        - `$\alpha_C$`
        - For later {$\beta$,Entropy}
    - Temporary Variables
        - V-Value of Critic Network 
            - Predicted `v_p`
            - Target    `v_t`
        - Advantage Value `a`
        - Loss 
            - Policy `l_p`
            - Critic `l_c`


## ONLINE-OFFLINE DILLEMA
- There are three versions of actor critic: Refer [#28](https://github.com/BlackDChase/Profit-Optimizer-A3C/issues/28)
    1. Online Sliding Window
    2. Online Episodic
    3. Offline


## Detailed View Of The Code
- Dataset
    - We have used a normalized dataset contaning the following coloumns :: Ontario Price, Ontario Demand, Ontario Supply,Northwest,Northeast,Ottawa,East,Toronto,Essa,Bruce, Northwest Nigiria, West.
    - This is used to train the Environment so that it closley mimicks a real Environment.
- Environment
    - Environment plays a central role in training of RL algorithms, an Environment (simulated or real ) is     required for proper interaction with the agent , change its state with respect to agents actions and most importantly give the agent a reward signal for its actions.
    - We use a LSTM based simulated Environment, Long short-term memory is an artificial recurrent neural network LSTM has feedback connections. It can not only process single data points, but also entire sequences of data.
        - Our LSTM based simulated Environment is trained on a normalized electricity demand-supply dataset. 
        - In order to ensure that the states predited by the simulated Environment stay within resonable limit as in a real Environment, we use a HARDTanh() activation function. 
        - After being trained on this dataset, the step() function of Environment is called by the agent. It recieves the action taken by the agent. Based on this action , LSTM Environment calculates a new observation (state) and also the reward for the actions of the agent.
        - Reward can be calculated by defining a MinAllowed and MaxAllowed limits. If the Agent goes beyond these two limits, the Environment heavily punishes the agent by creating a huge negative reward.(This is done because in a ideal scenario we must discourage the agent to set the price too high or too low, both of which are undesirable), then reward can be set as :: `(demand-supply) * price * correction`, where correction is a term used to penalize the agent if the price crosses the min-max extremities.
        - It can be easily verified by the reward formula that when `demand >= supply`, the agent must increase the price, doing so will result in a much higher reward (Desirable because elctricity company will make profit), whereas if `demand <= Supply`, the price must be lowerd , so that demand increases and Loss of the electricity company is minimized. 
        - reset() method of the Environment gives a state to the agent from which it must start exploring the  Environment.
- Agent
    - The Agent is a A3C Reinforcement learning algorithm , which tries to predict for each state , the best possible action that can be taken(i.e increasing the price or decreasing it by a certain amount)
    - The State is the 13 coloumns described in the dataset. The action space can be varied in size and each action is the percentage change in the current price (Positive or negative)
    - Agents can be divided in two parts
        - Master :: This is the supervising Agent and the one used in realtime learning, it is responsible for the creation of Policy and Critic Network, The Slave agents and coordinating them. In the online phase it takes the actual action in realtime and updats the networks according to A2C alsorithm.
            - Policy Net :: A neural net which acts as a function approximator to the ideal policy, it takes as input the current state of the agent, and outputs the Probability of each of the actions, which can then be sampled. Useful for predictiong the next action which agent must take.
            - Critic Net :: A neural net which acts as a function approximator to the ideal value of a state, the higher the Value of state the better it is.Input is the State and Output is the Value V(s) of the state. useful for calculating the Advantage, which is then fed to the Actor for its policy gradient.
        - Slave :: These Agents are created ast training time for efficient exploration of the environment. They gather the trajectory, caclulate the advantage,loss and perform policy gradient for updation of policy Network and also update the critic network asynchronously.  


## Conclusion
- Our model is a a3c based agent, which has two different types of agents GOD(Master) and BOSS(Slave).
  God is responsible for taking the final action in a Online setting as well as coordination of
  various boss agents and the network.
  Boss is responsible for exploration of the environment and applying a2c algorithm, different boss agents
  then asynchronously update the Network.
- Model is concerned with choosing the appropriate price increase/decrease for the current state(current demand)
  in hope that it will maximize the profit of the energy company and minimize the losses.


## Annexure
`@` Overriden method
`%` Semaphore required
