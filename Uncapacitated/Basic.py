import numpy as np
from tqdm import tqdm
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import Customers
import QLearner

#set seed
np.random.seed(11)

#number of prices we will offer
n_prices = 100

#number of customers each day
n_cust = 100

#number of episodes
n_episodes = 10**5

#explore probability for e-Greedy algorithm
explore_prob = .1

#initialise q learner
qlearner = QLearner.QLearner()

#number of prices we can set as actions
qlearner.setNActions(n_prices+1)

#learning rate/step size of algorithm
qlearner.setLearnRate(0.1)

#discount factor = 0 as no definition of new state
qlearner.setDiscountFactor(0)

#initialise q table
qlearner.initialiseQTable()

#initialise a cutomer
customer = Customers.Customer()

for episode in tqdm(range(n_episodes)):
    #set up reward for today
    current_reward = 0

    #choose to explore or exploit
    if np.random.uniform() < explore_prob:
        current_action =  int(np.random.randint(n_prices))
    else:
        current_action = qlearner.getBestActionInt()
        

    #for each customer
    for i in range(n_cust):
        #generate customer threshold
        customer.setThresholdLinear()

        #if the action is less than the customers willingness to pay
        #reward the agent
        if current_action < customer.getThreshold():
            current_reward += current_action
        #else they get nothing
            
    final_reward = current_reward/n_cust

    qlearner.QLearningUpdate(action = current_action, reward = final_reward)

np.save("qtables/basic", qlearner.getQTable())