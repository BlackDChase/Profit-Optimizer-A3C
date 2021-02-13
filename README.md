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

# State of understanding at 14-02-21
## Agent-Environment Dependence
- Agent will only maximize REWARD which it'll recieve from Env and nothing else. It's the duty of env. to give reward signal in a way which forces agent to learn to maximize the actual Profit Parameter ( Indirectly)
## Birds Eye view of Architecture:
- Action space can be discrete collection of the individual price increase like  [-5,-4,-3,-2,-1,0,1,2,3,4,5]... etc. each element 'x' corresponds to adding 'x' to the current price.
- The state can be  [Current Supply, Current Demand , Current Price ] , or a 2d colllection of [Current Supply, Current Demand , Current Price ] for a duration (like a entire week) as Black suggested ( Exact thing is to be decided) , the latter approach can increase the state space exponentially.
- A2C can be implemented as , a parametrized policy network (a DL based model , mapping state ->action probabilities) , a agent class which tries to maximize the reward through advantage policy gradient and a critic which supplies the actor with the advantage information. There's also Environment which provides the actor and the critic with the reward for a particular action.
