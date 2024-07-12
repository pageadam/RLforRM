import numpy as np
from matplotlib import pyplot as plt

#number of episodes
n_episodes = 10**5

reward_norm = np.load("rewards/QLearnerReward.npy")
reward_cross = np.load("rewards/QLearnerReward.npy")
reward_back = np.load("rewards/BackLearnerReward.npy")
reward_crossback = np.load("rewards/QLearnerReward.npy")

av_reward_norm = np.cumsum( [ np.mean(reward_norm[episode,:]) for episode in range(n_episodes) ] )
av_reward_cross = np.cumsum( [ np.mean(reward_cross[episode,:]) for episode in range(n_episodes) ] )
av_reward_back = np.cumsum( [ np.mean(reward_back[episode,:]) for episode in range(n_episodes) ] )
av_reward_crossback = np.cumsum( [ np.mean(reward_crossback[episode,:]) for episode in range(n_episodes) ] )

plt.plot(av_reward_norm,label ='Q-Learner')
plt.plot(av_reward_cross,label = 'Cross-Learner')
plt.plot(av_reward_back,label = 'Back-Learner')
plt.plot(av_reward_crossback,label = 'BackAug-Learner')
plt.xlabel("episode")
plt.ylabel("reward")
plt.title("Cross Learner for Q-learning Pricing")
plt.legend()
plt.savefig("plots/rewards.png")