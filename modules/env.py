import gym
import pandas as pd
import numpy as np
from collections import deque
import random
import log
import multiprocessing

__author__ = 'Biribiri,BlackDChase'
__version__ = '0.3.8'

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
        random_index = random.randint(0, len(self.df) - self.max_input_len + 1)
        self.first_input = self.df.iloc[random_index:random_index + self.max_input_len, :].values
        return self.first_input

class LSTMEnv(gym.Env):
    def __init__(self,
                 model,
                 dataset_path="../datasets/normalized_weird_13_columns_with_supply.csv",
                 min_max_values_csv_file="../datasets/min_max_values_13_columns_with_supply.csv",
                 max_input_len=25,
                 actionSpace=[-15,-10,0,10,15],
                 debug=False):
        """
        model = trained LSTM model from lstm.py

        By default, all the data (including the dataset helper's dataset) are
        all normalized.
        Only when sending data to the agent, is the data de-normalized before
        sending it. This includes second-order data sent to the agent such as
        reward.
        """
        self.model = model
        self.dataset_helper = DatasetHelper(dataset_path,max_input_len)

        # Initialize self.observation_space
        # required for self.current_observation
        self.observation_space = gym.spaces.Box(low=-np.inf,high=np.inf,shape=(self.model.input_dim,))

        # create model input deque
        self.model_input = deque([], maxlen=max_input_len)
        self.max_input_len = max_input_len
        self.actionSpace=actionSpace

        # list of min / max values for each of the 13 columns used for wrapping inputs and unwrapping outputs
        self.min_max_values = pd.read_csv(min_max_values_csv_file)

        self.debug=debug

    def reset(self):
        """
        Return a value within self.observation_space
        Why clear input?
        """
        self.model_input.clear()
        dataset_helper_input = self.dataset_helper.reset()
        [self.model_input.append(element) for element in dataset_helper_input]

        # convert to numpy version
        np_model_input = np.array(self.model_input)

        """
        Saving starting states for testing
        """
        self.startState = []
        [self.startState.append(element) for element in dataset_helper_input]

        curr = multiprocessing.current_process()
        if self.debug:
            log.debug(f"Reset call for {curr.name}")
        current_observation = self.model.forward(np_model_input, numpy=True)

        self.current_observation = current_observation
        self.denormalized_current_observation = self.denormalize(self.current_observation)
        if self.debug:
            log.debug(f"Reset complete for {curr.name}")
            log.debug(f">current_observation = {self.current_observation}")
            log.debug(f">denormalized_current_observation = {self.denormalized_current_observation}")
        return self.current_observation

    # TODO What is this function for
    def possibleState(self,time=100):
        """
        Output  : Return the output of the enviornment for `time` number of steps without the feedback of A3C agent.
        """
        states = []
        model_input = deque([], maxlen=self.max_input_len)
        [model_input.append(element) for element in self.startState]

        for i in range(time+1):
            np_model_input = np.array(model_input)
            current_observation = self.model.forward(np_model_input, numpy=True)
            model_input.append(current_observation)
            current_observation = self.denormalize(current_observation)
            log.info(f"Possible set {i} = {current_observation}")
            states.append(current_observation)
        return np.array(states)

    def step(self, action):
        """
        Calculate new current observation
        Calculate reward
        Return relevant data
        """
        if action < 0 or action >= len(self.actionSpace):
            log.info(f"Illegal action = {action}")
            log.debug(f"Action Space = {self.actionSpace}")
            import sys
            sys.exit()

        # Set done as False as there's no reason to end an episode
        done = False

        # Implement effects of action
        new_price = self.get_new_price(action)
        new_price_denormalized = self.get_new_price(action, denormalize=True)

        # get reward
        # Ensure that numpy array shape is (1,), not () otherwise conversion to torch.Tensor will get messed up
        # Use denormalized new price to get denormalized reward
        denormalized_reward = np.array([self.get_reward(new_price_denormalized, denormalize=True)])

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
        self.denormalized_current_observation = self.denormalize(self.current_observation)

        if self.debug:
            log.debug(f">current_observation = {self.current_observation}")
            log.debug(f">denormalized_current_observation = {self.denormalized_current_observation}")

        return self.current_observation, denormalized_reward, done, {}

    def get_new_price(self, action, denormalize=False):
        """
        Modify the price by a percentage according to the action taken.
        """
        price_index = 0
        if denormalize:
            try:
                old_price = self.denormalized_current_observation[price_index]
            except:
                self.denormalized_current_observation = self.denormalize(self.current_observation)
                old_price = self.denormalized_current_observation[price_index]
        else:
            old_price = self.current_observation[price_index]
        # Increase or decrease the old price by a percentage, as defined by actions
        new_price = old_price * (1 + self.actionSpace[action] / 100)
        return new_price

    def get_reward(self, new_price, denormalize=False):
        """
        Calculate reward based on the new_price

        Demand is always positive in the dataset
        Prices can be negative in the dataset

        We cannot use min(demand, supply) * price because it does not take into
        account times when supply is greater than demand, meaning the
        electricity would either be sent at a loss, or would be bought from
        these smaller producers.
        """
        price_index = 0
        ontario_demand_index = 1
        supply_index = 2
        if denormalize:
            self.denormalized_current_observation = self.denormalize(self.current_observation)

            log.debug(f"self.denormalized_current_observation.shape = {self.denormalized_current_observation.shape}")
            demand = self.denormalized_current_observation[ontario_demand_index]
            supply = self.denormalized_current_observation[supply_index]
        else:
            log.debug(f"self.current_observation.shape = {self.current_observation.shape}")
            demand = self.current_observation[ontario_demand_index]
            supply = self.current_observation[supply_index]


        """
        Acc to dataset minAllowed is 0, maxAllowed is arround 2.3k.
        Considering three cases:
        abs(Price) ∉ (minAllowed,maxAllowed): Heavily Punished by High Correction Value
        Demand - Supply, Price same sign    : Profit, Rewarded if correction is positive, punished otherwise
        Demand - Supply, Price opposite sign: Loss, punished
        Decreaseing the overall Reward value: Correction/=(10**6)
        """
        correction = self.min_max_values["max"][price_index] - abs(new_price)
        if correction>0:
            correction/=self.min_max_values["max"][price_index]
        if ((demand-supply) * new_price<0 )&(correction<0):
            correction=-correction
        correction/=(10**6)
        log.info(f"State set = {new_price}, {correction}, {demand}, {supply}")
        return (demand - supply) * new_price * correction

    def denormalize(self, array):
        """
        Take any numpy array of 13 elements and de-normalize it, that is, undo
        the normalization done to the data, and then return it.
        """
        for feature in range(array.shape[0]):
            minv = self.min_max_values["min"][feature]
            maxv = self.min_max_values["max"][feature]
            value = array[feature]
            array[feature] = (value * (maxv - minv)) + minv - 1
        return array

    def normalize(self, array):
        """
        Take any numpy array of 13 elements and normalize it according to
        pre-defined values.
        """
        for feature in range(array.shape[0]):
            minv = self.min_max_values["min"][feature]
            maxv = self.min_max_values["max"][feature]
            value = array[feature]
            array[feature] = (value - minv + 1)/(maxv - minv)
        return array
