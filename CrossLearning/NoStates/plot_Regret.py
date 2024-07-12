import numpy as np
from matplotlib import pyplot as plt
import scipy.stats

#number of episodes
n_episodes = 10**5

#number of macro replications
n_macro_reps = 30

confidence_level = 0.975

def F_tau(P,MaxP):
    y = [MaxP - i for i in range(MaxP+1)]
    fprobs = np.cumsum([i/sum(y) for i in y])
    return fprobs[P]

maxprice=100
Ft = []
St=[]
Et = []
for p in range(maxprice+1):
    #Ft.append((p*(2*M-1-p))/(M*(M-1)))
    Ft.append(F_tau(p,maxprice))
    St.append(1-Ft[p])
    Et.append(St[p]*p)

reward_norm = np.load("rewards/QLearnerReward.npy")
reward_cross = np.load("rewards/CrossLearnerReward.npy")
reward_back = np.load("rewards/BackLearnerReward.npy")
reward_crossback = np.load("rewards/CrossBackLearnerReward.npy")

true_mean_reward = [i*max(Et) for i in range(n_episodes)]
regret_norm = np.zeros((n_episodes,n_macro_reps))
regret_cross = np.zeros((n_episodes,n_macro_reps))
regret_back = np.zeros((n_episodes,n_macro_reps))
regret_crossback = np.zeros((n_episodes,n_macro_reps))

av_reward_norm = np.cumsum( [ np.mean(reward_norm[episode,:]) for episode in range(n_episodes) ] )
av_reward_cross = np.cumsum( [ np.mean(reward_cross[episode,:]) for episode in range(n_episodes) ] )
av_reward_back = np.cumsum( [ np.mean(reward_back[episode,:]) for episode in range(n_episodes) ] )
av_reward_crossback = np.cumsum( [ np.mean(reward_crossback[episode,:]) for episode in range(n_episodes) ] )


for rep in range(n_macro_reps):
    for ep in range(n_episodes):
        regret_norm[ep,rep] = max(Et) - reward_norm[ep,rep]
        regret_cross[ep,rep] = max(Et) - reward_cross[ep,rep]
        regret_back[ep,rep] = max(Et) - reward_back[ep,rep]
        regret_crossback[ep,rep] = max(Et) - reward_crossback[ep,rep]

    regret_norm[:,rep] = np.cumsum(regret_norm[:,rep])
    regret_cross[:,rep] = np.cumsum(regret_cross[:,rep])
    regret_back[:,rep] = np.cumsum(regret_back[:,rep])
    regret_crossback[:,rep] = np.cumsum(regret_crossback[:,rep])

av_reg_norm = [np.mean(regret_norm[episode,:]) for episode in range(n_episodes)]
av_reg_cross = [np.mean(regret_cross[episode,:]) for episode in range(n_episodes)]
av_reg_back = [np.mean(regret_back[episode,:]) for episode in range(n_episodes)]
av_reg_crossback = [np.mean(regret_crossback[episode,:]) for episode in range(n_episodes)]


var_reg_norm = [np.var(regret_norm[episode,:]) for episode in range(n_episodes)]
var_reg_cross = [np.var(regret_cross[episode,:]) for episode in range(n_episodes)]
var_reg_back = [np.var(regret_back[episode,:]) for episode in range(n_episodes)]
var_reg_crossback = [np.var(regret_crossback[episode,:]) for episode in range(n_episodes)]

zvalue = scipy.stats.norm.ppf(confidence_level)

CI_upper_norm = [av_reg_norm[episode] + zvalue*((var_reg_norm[episode]/n_macro_reps)**(1/2)) for episode in range(n_episodes)]
CI_lower_norm = [av_reg_norm[episode] - zvalue*((var_reg_norm[episode]/n_macro_reps)**(1/2)) for episode in range(n_episodes)]

CI_upper_cross = [av_reg_cross[episode] + zvalue*((var_reg_cross[episode]/n_macro_reps)**(1/2)) for episode in range(n_episodes)]
CI_lower_cross = [av_reg_cross[episode] - zvalue*((var_reg_cross[episode]/n_macro_reps)**(1/2)) for episode in range(n_episodes)]

CI_upper_back = [av_reg_back[episode] + zvalue*((var_reg_back[episode]/n_macro_reps)**(1/2)) for episode in range(n_episodes)]
CI_lower_back = [av_reg_back[episode] - zvalue*((var_reg_back[episode]/n_macro_reps)**(1/2)) for episode in range(n_episodes)]

CI_upper_crossback = [av_reg_crossback[episode] + zvalue*((var_reg_crossback[episode]/n_macro_reps)**(1/2)) for episode in range(n_episodes)]
CI_lower_crossback = [av_reg_crossback[episode] - zvalue*((var_reg_crossback[episode]/n_macro_reps)**(1/2)) for episode in range(n_episodes)]

plt.plot(av_reg_norm,label ='Q-Learner',color= 'blue')
plt.fill_between(range(n_episodes),CI_lower_norm,CI_upper_norm,color= 'blue',alpha=0.1,label = '95% CI Q')
plt.plot(av_reg_cross,label = 'Cross-Learner',color='orange')
plt.fill_between(range(n_episodes),CI_lower_cross,CI_upper_cross,color='orange',alpha=0.1,label = '95% CI Cross')
plt.plot(av_reg_back,label = 'Back-Learner',color = 'green')
plt.fill_between(range(n_episodes),CI_lower_back,CI_upper_back,color = 'green',alpha=0.1,label = '95% CI Back')
plt.plot(av_reg_crossback,label = 'CrossBack-Learner',color = 'red')
plt.fill_between(range(n_episodes),CI_lower_crossback,CI_upper_crossback,color = 'red',alpha=0.1,label = '95% CI BackCross')
plt.xlabel("episode")
plt.ylabel("mean regret")
plt.title("Cross Learner for Q-learning Pricing")
plt.legend()
plt.savefig("plots/regret.png")