# Adjusting Price to maximize profit using A3C
## Decisions
- Model has to be scaled in way that it can adjust for the following scenario
- Need to decide whether to use `reducedDF` or `advancedDF`
- Which Demand parameter to be used `Market Demand` or `Ontorio Demand`
- `Profit parameter` Should be of a time frame that is to be maximized
    - Can be `Demand*Price`
    - If it has to optimise profit for a longer term than t should be predicting for a longer term.
- __Environment is going to produce a reward__

## Bird Eye View
- God Agent:
    - Will take final action
    - A supervising psuedo agent that takes action
    - Methods
        - `__init__`
        - `Get Current State`
        - `Take Action`
        - `Initiate Boss agents`
        - `Train Boss agents`
    - Attributes
        - `%Policy Network`
        - `%Critic Network`
        - `Max Episode`
        - Trajectory Lenght `T`
        - `Number of boss agents`
        - `List of Boss agents`
        - `State`
        - `Price`
- Boss Agent(God Agent):
    - Will Update the network
    - Methods
        - `__init__`
        - `train`
        - `Gather and Store` $(s_t,a_t,r_t,s_t\prime)$
        - Calculate `v_p`
        - Calculate `v_t`
        - Calculate `gae`
        - Calculate and Update `l_p %`
        - Calculate and Update `l_c %`
    - Attributes
        - $\alpha_A$
        - $\alpha_C$
        - $\beta$
    - Temporary Variables
        - V-Value of Critic Network
            - Predicted `v_p`
            - Target    `v_t`
        - Advantage Value `a`
        - Loss 
            - Policy `l_p`
            - Critic `l_c`
## Annexure
@ Overriden method
% Semaphore required
