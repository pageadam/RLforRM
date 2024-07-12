import matplotlib.pyplot as plt
import numpy as np

n_lead_time = 7
n_prices = 100

Exp_reward = []
St_p=[]
yf_lead = []
for lead in range(n_lead_time+1):

    f = (n_lead_time - lead)/n_lead_time

    x=range(n_prices)
    y = [n_prices-i for i in x]
    yf_lead.append([f*i for i in y])
    y = [i/sum(y) for i in y]

    St_lead_p = []
    Exp_reward_lead_p = []

    for i in range(n_prices):
        St_lead_p.append(f*sum(y[i:n_prices]))
        Exp_reward_lead_p.append(St_lead_p[i]*(i))

    St_p.append(St_lead_p)
    Exp_reward.append(Exp_reward_lead_p)
    yf_lead[lead].reverse()

q_table = np.load("qtables/lead.npy")

col = [None, 'red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
for lead in range(1,n_lead_time):
    plt.plot(q_table[lead,:],'o',color=col[lead])
plt.xlabel("price, £")
plt.ylabel("Q-Value")
plt.title("Q-Values of infinite capacity, booking with lead time")
plt.legend(['Lead = 1','Lead = 2','Lead = 3','Lead = 4','Lead = 5','Lead = 6','Lead = 7'])
for lead in range(1,n_lead_time+1):
    plt.plot(yf_lead[lead],Exp_reward[lead],color=col[lead],linestyle = 'dotted')
plt.savefig("plots/lead.pdf")
plt.savefig("plots/lead.png")