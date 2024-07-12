import matplotlib.pyplot as plt
import numpy as np

maxprice=100

def F_tau(P,MaxP):
    y = [MaxP - i for i in range(MaxP+1)]
    fprobs = np.cumsum([i/sum(y) for i in y])
    return fprobs[P]


Ft = []
St=[]
Et = []
for p in range(maxprice+1):
    Ft.append(F_tau(p,maxprice))
    St.append(1-Ft[p])
    Et.append(St[p]*p)

q_table = np.load("qtables/basic.npy")

plt.plot(q_table,'o',color = (42/255,49/255,80/255),label = 'Q-Learner')
plt.plot(Et,linestyle = 'dotted',color=(255/255,145/255,129/255), label = 'Exepcted Reward')
plt.xlabel("price, £")
plt.ylabel("Q-Value")
plt.title("Q-values of infinite capacity, booking on arrival")
plt.legend()
plt.savefig("plots/basic.pdf")
plt.savefig("plots/basic.png")