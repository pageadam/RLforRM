import numpy as np
from matplotlib import pyplot as plt


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

q_table = np.load("qtables/QLearner.npy")
q_table_cross = np.load("qtables/CrossLearner.npy")
q_table_back = np.load("qtables/BackLearner.npy")
q_table_crossback = np.load("qtables/CrossBackLearner.npy")

plt.plot(q_table,'o',label ='Q-Learner')
plt.plot(q_table_cross,'o',label = 'Cross-Learner')
plt.plot(q_table_back,'o',label = 'Back-Learner')
plt.plot(q_table_crossback,'o',label = 'CrossBack-Learner')
plt.plot(Et,color='r',linestyle = 'dotted')
plt.xlabel("Price, £")
plt.ylabel("Q-Value")
plt.title("Cross Learner for Q-learning Pricing")
plt.legend()
plt.savefig("plots/qTables.png")