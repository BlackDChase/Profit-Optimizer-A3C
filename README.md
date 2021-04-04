# Adjusting Price to maximize profit using A3C
## Decisions
- Model has to be scaled in way that it can adjust for the following scenario
- Need to decide whether to use `reducedDF` or `advancedDF`
- Which Demand parameter to be used `Market Demand` or `Ontorio Demand`
- `Profit parameter` Should be of a time frame that is to be maximized
    - Can be `Demand*Price`
    - If it has to optimise profit for a longer term than t should be predicting for a longer term.
- __Environment is going to produce a reward__, The only thing matters for the agent is the reward , so it doesn't matters to agent how that reward is distributed, it will only chase the state which gives high reward. Now a preliminary reward can be (price * demand) , this needs to be refined later

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
- There are two versions of actor critic:
    1. Batch
    2. Online
        - It seems that Online one is more suited for our problems.
        - In online we don't need to create trajectory.
        - We just sample a action and update the network step-by-step.

## Conclusion
- Our model is a a3c based agent, which has two different types of agents GOD and BOSS.
  God is responsible for taking the final action in a Online setting as well as coordination of
  various boss agents and the network.
  Boss is responsible for exploration of the environment and applying a2c algorithm, different boss agents
  then asynchronously update the Network.
- Model is concerned with choosing the appropriate price increase/decrease for the current state(current demand)
  in hope that it will maximize the profit of the energy company and minimize the losses.

## Annexure
@ Overriden method
% Semaphore required
