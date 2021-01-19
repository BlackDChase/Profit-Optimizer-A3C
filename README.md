# Adjusting Price to maximize profit using A3C
## Decisions
- Model has to be scaled in way that it can adjust for the following scenario
- Need to decide whether to use `reducedDF` or `advancedDF`
- Which Demand parameter to be used `Market Demand` or `Ontorio Demand`
- Electricity consumpution of next time frame to be forecasted to name set a price
    - `Time Frame` can be of 1 day or a week
    - `Profit parameter` Should be of that time frame that is to be maximized
- Profit parameter to be maximised
    - Can be `Demand*Price`
- __Environment is going to produce a reward__
