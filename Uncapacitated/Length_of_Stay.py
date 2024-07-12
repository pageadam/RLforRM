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
n_prices = (100,100)

#number of customers each day
n_cust = 100

#number of episodes
n_episodes = 10**7

#explore probability for e-Greedy algorithm
explore_prob = .1

#maximum length of stay a customer can stay for
n_length_of_stay = 2

#initialise q learner
qlearner = QLearner.QLearner()

#number of prices we can set as actions
qlearner.setNActions(n_prices)

#learning rate/step size of algorithm
qlearner.setLearnRate(0.1)

#discount factor = 0 as no definition of new state
qlearner.setDiscountFactor(0)

#initialise q table
qlearner.initialiseQTable()

#initialise a cutomer
customer = Customers.Customer()

current_action = [0]*n_length_of_stay

for episode in tqdm(range(n_episodes)):
    
    if np.random.uniform() < explore_prob:
        current_action =  tuple(np.random.randint(n_prices[1],size = n_length_of_stay))
    else:
        current_action = qlearner.getBestAction()

    #set up reward for today
    current_reward = 0

    #for each customer
    for i in range(n_cust):
        #generate customer threshold
        customer.setThresholdLinear()
        #generate customer length of stay
        customer.setLoSUniform(min_LoS=1,max_LoS=2)
        #calculate total threshold dependent on LoS and price per day willingness to pay
        customer.setThresholdExact(customer.getLoS()*customer.getThreshold())

        #if the action is less than the customers willingness to pay
        #reward the agent
        if sum(current_action[:customer.getLoS()]) < customer.getThreshold():
            current_reward += sum(current_action[:customer.getLoS()])
        #else they get nothing

    
    final_reward = current_reward/n_cust

    qlearner.QLearningUpdate(action = current_action, reward = final_reward)

np.save("qtables/LoS", qlearner.getQTable())