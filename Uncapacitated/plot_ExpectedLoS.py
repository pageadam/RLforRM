import matplotlib.pyplot as plt
import numpy as np

n_prices = (100,100)

def F_tau(P,MaxP):
    y = [MaxP - i for i in range(MaxP+1)]
    fprobs = np.cumsum([i/sum(y) for i in y])
    return fprobs[P]


Exp_Reward_LoS = np.zeros((n_prices[0]+1,n_prices[1]+1))
probs = [.5,.5]
for p1 in range(n_prices[0]+1):
    for p2 in range(n_prices[1]+1):
        Exp_Reward_LoS[p1,p2] = probs[0]*p1*(1-F_tau(p1,n_prices[0])) + probs[1]*(p1+p2)*(1-F_tau(int((p1+p2)/2),n_prices[1]))

plt.imshow(Exp_Reward_LoS, cmap='hot',origin='lower')
plt.xlabel("Day 2 Price, £")
plt.ylabel("Day 1 Price, £")
plt.colorbar(label = "Expected Reward")
plt.title("True Expected Reward Value")
plt.savefig("plots/ExpectedLoS.pdf")
plt.savefig("plots/ExpectedLoS.png")