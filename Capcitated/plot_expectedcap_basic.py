import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

max_cap = 100
n_prices = 100
gamma = 1

def F_tau(P,MaxP):
    return (P*(2*MaxP-1-P))/(MaxP*(MaxP-1))

def Create_Rew_Prob_ERew_Matrix(ns,na,N):

    reward_matrix = np.zeros((ns+1,na+1,ns+1))

    trans_prob_mat = np.zeros((ns+1,na+1,ns+1))

    exp_val_mat = np.zeros((ns+1,na+1))

    for s in tqdm(range(1,ns+1)):
        
        for a in range(0,na+1):

            exp_val_hold = []

            for sprime in range(1,s+1):
                
                sold = min(s-sprime,N)

                rew=(sold*a/N)
                reward_matrix[s,a,sprime] = rew

                prob = math.comb(N,sold)*(F_tau(a,na)**(N-sold))*((1-F_tau(a,na))**sold)
                if prob > 1 or prob < 0:
                    print(prob)
                trans_prob_mat[s,a,sprime] = prob
                
                
                exp_val_hold.append(prob*rew)
            
            sold = s

            rew = (sold*a/N)
            reward_matrix[s,a,0] = rew

            prob = 1 - sum(trans_prob_mat[s,a,:])
            trans_prob_mat[s,a,0] = prob

            # if prob > 1 or prob < 0:
            #         print(prob,s,a)

            exp_val_hold.append(prob*rew)
            
            exp_val_mat[s,a] = sum(exp_val_hold)

    return reward_matrix, trans_prob_mat, exp_val_mat

R,P,ER = Create_Rew_Prob_ERew_Matrix(100,100,100)

best_price = []

for s in range(max_cap+1):
    best_price.append(np.argmax(ER[s,:]))

true_table = np.zeros((max_cap+1,n_prices+1))

for s in range(1,max_cap+1):
    for a in range(n_prices+1):
        true_table[s,a] = ER[s,a] + gamma*sum([ P[s,a,sprime] * np.max(true_table[sprime,:]) for sprime in range(0,s+1)])

plt.imshow(true_table, cmap='hot',aspect='auto')
plt.title("True Expected Return")
plt.xlabel("Price, £")
plt.ylabel("Current No. of spaces available")
plt.colorbar(label="Q-Value")
plt.savefig("plots/expected_cap_basic.pdf")
plt.savefig("plots/expected_cap_basic.png")