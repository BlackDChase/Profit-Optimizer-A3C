import gym
import pandas as pd
import numpy as np
from collections import deque
import random
import log

# increment / decrement price by x%
ACTION_INC_10 = 0
ACTION_INC_50 = 1
ACTION_DEC_10 = 2
ACTION_DEC_50 = 3
ACTION_HOLD = 4

class DatasetHelper:
    def __init__(self, dataset_path, max_input_len):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(self.dataset_path)
        self.max_input_len = max_input_len

    def reset(self):
        """
        Choose a random starting timestep for gym resets from dataframe
        Return it as a numpy array
        """
        random_index = random.randint(0, len(self.df) - max_input_len + 1)
        self.first_input = self.df.iloc[random_index:random_index + max_input_len + 1, :].values
        return self.first_input

class LSTMEnv(gym.Env):
    def __init__(self, model, dataset_path, max_input_len=25):
        """
        model = trained LSTM model from lstm.py
        """
        self.model = model
        self.dataset_helper = DatasetHelper(dataset_path,max_input_len)

        # Initialize self.observation_space
        # required for self.current_observation
        self.observation_space = gym.spaces.Box(low=-np.inf,high=np.inf,shape=(self.model.input_dim))

        # create model input deque
        model_input = deque([], maxlen=max_input_len)

        # set self.current_observation
        self.reset()

    def reset(self):
        """
        Return a value within self.observation_space
        """
        self.model_input.clear()
        self.model_input.append(self.datahelper.reset())

        # convert to numpy version
        np_model_input = np.array(self.model_input)

        self.current_observation = self.model.forward(np_model_input, numpy=True)
        return self.current_observation

    def step(self, action):
        """
        Calculate new current observation
        Calculate reward
        Return relevant data
        """
        if action not in [0, 1, 2, 3, 4]:
            print("Illegal action")
            log.debug(f"action = {action}")
            import sys
            sys.exit()

        # Set done as False as there's no reason to end an episode
        done = False

        # Implement effects of action
        new_price = self.get_new_price(action)

        # get reward
        reward = self.get_reward(new_price)

        # We update the price in the current observation
        # This ensures that the model takes into account the action we just
        # took when giving us the next timestep
        price_index = 0
        self.current_observation[price_index] = new_price

        # append the current observation to the model input
        self.model_input.append(self.current_observation)

        # get the next observation
        numpy_model_input = np.array(self.model_input)
        self.current_observation = self.model.forward(numpy_model_input, numpy=True)

        return self.current_observation, reward, done, {}

    def get_new_price(self, action):
        """
        Modify the price according to the action taken.
        10,15 % change rather than 10,50
        """
        price_index = 0
        old_price = self.current_observation[price_index]
        if action == ACTION_HOLD:
            return old_price
        elif action == ACTION_INC_10:
            return old_price * 1.1
        elif action == ACTION_INC_50:
            return old_price * 1.15
        elif action == ACTION_DEC_10:
            return old_price * 0.9
        elif action == ACTION_DEC_50:
            return old_price * 0.85
        else:
            print("WARNING: Illegal action")
            log.debug(f"action = {action}")
            # immediately exit
            import sys
            sys.exit()

    def get_reward(self, new_price):
        """
        Calculate reward based on the new_price
        """
        market_demand_index = 1
        ontario_demand_index = 2
        demand = (self.current_observation[market_demand_index]
        + self.current_observation[ontario_demand_index])/2
        return (demand * new_price)**(1/2)
