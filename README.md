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
        - `Get Current State`
        - `Take Action`
        - `Initiate Boss agents`
        -
    - Attributes
        - `$Policy Network`
        - `$Critic Network`
        - `Max Episode`
        - Trajectory Lenght `T`
        - `Number of boss agents`
        - `List of Boss agents`
- Boss Agent:
    - Will Update the network
    - Methods
    - Attributes

## Annexure
@ Overriden method
$ Semaphore protected attributes
