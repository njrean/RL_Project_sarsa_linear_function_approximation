import itertools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import pickle
from lib.plotting import EpisodeStats
import gym
import os
import re

if "../" not in sys.path:
  sys.path.append("../") 

from lib import plotting

matplotlib.style.use('ggplot')
from sklearn.linear_model import SGDRegressor

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Estimator():
    """
    Class for being a state action-value approximation function
    """
    
    def __init__(self, env, scaler=None, featurizer=None, mode='new'):
        # Create a separate model for each action in the environment's
        self.models = []

        #need to create new estimator
        if mode == 'new':
            self.scaler = scaler
            self.featurizer = featurizer

            #initialize the models
            for _ in range(env.action_space.n): # env.action_space.n of mountain car is equal to 3
                model = SGDRegressor(learning_rate="constant")
                model.partial_fit([self.featurize_state(env.reset())], [0])
                self.models.append(model)

        #need to load trained weight
        elif mode == 'load':
            self.load_weight()
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform([state[0]])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]
    
    def predict(self, s, a=None):
        """
        Makes value function predictions.
        
        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for
            
        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
        """
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]
    
    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])

    def save_weight(self, name='Sarsa_LinearSGD'):
        """
        Save weight of model, scaler and featurizer
        """
        #save weight of
        for i, m in enumerate(self.models):
            pickle.dump(m, open("{}_{}{}".format(name, i, '.pkl'), 'wb'))

        #save scaler
        pickle.dump(self.scaler, open('scaler.pkl', 'wb'))

        #save featurizer
        pickle.dump(self.featurizer, open('featurizer.pkl', 'wb'))

    def load_weight(self, name='Sarsa_LinearSGD'):
        """
        Load weight from saved weight
        """
        #load model
        self.models = []
        for i in range(3):
            with open("{}_{}{}".format(name, i, '.pkl'), 'rb') as file:
                m = pickle.load(file)
            self.models.append(m)

        #load scaler
        with open('scaler.pkl', 'rb') as file:
            self.scaler = pickle.load(file)

        #load featurizer 
        with open('featurizer.pkl', 'rb') as file:
            self.featurizer = pickle.load(file)

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
        which action has maximum value (get from approximation function) get more probability.
    """
    
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    
    return policy_fn

def sarsa(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0, save=False, save_history=1):
    """
    Train the action-value approximation function by using Sarsa algorithm to update weight in approximation function.

    Args:
        env: environment that use 
        estimator: An estimator that returns q values for a given state
        num_episodes: Number of episode that train and update weight in approximation function
        discount_factor: Discount factor that use in equation to update weight
        epsilon: The probability to select a random action . float between 0 and 1.
        save: Boolean to identify need to save model while training
        save_history: step to save model (use when save is True)

    Return:
        stat: The history of lenght and reward of each episode while training
    """

    # Create object EpisodeStats for collecting training history
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    for i_episode in range(num_episodes):
        
        policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
        state = env.reset()
        action_probs = policy(state)

        # random action from probabilitise of each action got from epsilon-greedy policy
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        for t in itertools.count():
            
            next_state, reward, end, _, _ = env.step(action)
            next_state = np.reshape(next_state, (1, -1))
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            q_values_next = estimator.predict(next_state)
            td_target = reward + discount_factor * q_values_next[next_action]
            
            estimator.update(state, action, td_target)
            
            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, reward), end="")
            
            # Break the loop when the agent reach terminal state.
            if end:
                break
                
            state = next_state
            action = next_action
        
        # Save model history
        if (i_episode+1)%save_history == 0 and save:
            name = "history/Sarsa_LinearSGD_iter"+str(i_episode+1)
            estimator.save_weight(name)
        
    return stats

def save_stat(stat, name='stat'):
    # Save stat history
    pickle.dump(stat,  open("{}{}".format(name, '.obj'), 'wb'))

def load_stat(name='stat.obj'):
    # Load saved stat history
    return pickle.load(open(name, 'rb'))

def num_sort(test_string):
    # Sort string in list that contain integer in string
    return list(map(int, re.findall(r'\d+', test_string)))[0]

def visualize_train(env, epsilon=0.1, history_path='./history'):
    """
    Visualize by rendering pygame window with saved model in history path
    """

    env = gym.make("MountainCar-v0", render_mode='human')
    
    model_history = os.listdir(history_path)
    model_history = set([f.split('_')[2] for f in model_history])
    model_history = sorted(model_history, key=num_sort)

    for iteration in model_history:
        name = "{}_{}".format("history/Sarsa_LinearSGD", iteration)
        estimator = Estimator(env, mode='load')
        estimator.load_weight(name)

        policy = make_epsilon_greedy_policy(estimator, epsilon, env.action_space.n)
        state = env.reset()
        env.render()

        total_reward = 0

        while(1):
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            state, reward, end, _, _ = env.step(action)
            state = state.reshape(1,-1)
            
            env.render()

            total_reward += reward

            print("\r{}: totla reward:{} action:{} state:{}".format(name, total_reward, action, state), end="")

            if end or total_reward<=-300:
                break