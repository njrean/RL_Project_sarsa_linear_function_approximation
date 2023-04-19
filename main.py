import gym
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from sarsa import Estimator, sarsa, save_stat, load_stat, visualize_train
import sys
from lib import plotting
import statistics

if "../" not in sys.path:
  sys.path.append("../") 

env = gym.make("MountainCar-v0")

#1.Feature Preprocessing: Normalize to zero mean and unit variance by some sample from environment
# Create scaler
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

#2.Converte a state to a featurizes represenation using RBF kernels with different variances to cover different parts of the space
# Create featurizer
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))

#3.1 Train model
#Create approximation function and train
estimator_sarsa = Estimator(env, scaler, featurizer)
stats_sarsa = sarsa(env, estimator_sarsa, 300, epsilon=0.0, save=True, save_history=30)
# Save weight and history of training
estimator_sarsa.save_weight()
save_stat(stats_sarsa)

#3.2 load model, scaler, feturizer and stat_history (if already has trained weight)
# estimator_sarsa = Estimator(env, mode='load')
# stats_sarsa = load_stat()

#4 Show statistic value and graphs of training 
print('Minimum step per episode:', min(stats_sarsa.episode_lengths))
print('Variance of step between episode 50-300:', statistics.stdev(stats_sarsa.episode_lengths[49:]))
print('Maximium reward:', max(stats_sarsa.episode_rewards))

plotting.plot_cost_to_go_mountain_car(env, estimator_sarsa)
plotting.plot_episode_stats(stats_sarsa, smoothing_window=25)

#5 Visualize trianed model in environment from saved weight.
visualize_train(env)


