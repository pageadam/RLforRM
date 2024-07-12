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
n_episodes = 10**7

#explore probability for e-Greedy algorithm
explore_prob = .1

#maximum capacity fo the car park
max_cap = 100

#initialise q learner
qlearner = QLearner.QLearner()

#define number  of states as maximum capacity
qlearner.setNStates(max_cap+1)

#number of prices we can set as actions
qlearner.setNActions(n_prices+1)

#learning rate/step size of algorithm
qlearner.setLearnRate(0.1)

#discount factor = 0 as no definition of new state
qlearner.setDiscountFactor(1)

#initialise q table
qlearner.initialiseQTable()

#initialise a cutomer
customer = Customers.Customer()

for episode in tqdm(range(n_episodes)):

    #Random starting state    
    current_capacity = np.random.randint(0,max_cap)
    qlearner.setCurrentState(current_capacity)

    #set up reward for today
    current_reward = 0

    #choose to explore or exploit
    if np.random.uniform() < 1: #explore_prob
        current_action =  int(np.random.randint(n_prices+1))
    else:
        current_action = qlearner.getBestActionInt(current_capacity)
        
    #craete new counter for new capacity
    new_capacity = current_capacity

    #for each customer
    for i in range(n_cust):
        #generate customer threshold
        customer.setThresholdLinear()

        #if the action is less than the customers willingness to pay
        #and there is enoough space for the customer
        #reward the agent
        if current_action < customer.getThreshold() and new_capacity < max_cap:
            current_reward += current_action
            new_capacity += 1 

            if new_capacity == max_cap:
                #if the car park is full just break the loop and move on to updating
                break  

        #else they get nothing
            
    final_reward = current_reward/n_cust

    qlearner.setNewState(new_capacity)
    qlearner.QLearningUpdate(action = current_action, reward = final_reward)

np.save("qtables/cap_basic", qlearner.getQTable())