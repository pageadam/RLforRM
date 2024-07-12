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
n_cust = 100*5

#number of episodes
n_episodes = 10**7

#explore probability for e-Greedy algorithm
explore_prob = 1

#maximum lead time a customer can arrive with
n_lead_time = 7

#maximum capacity fo the car park
max_cap = 100

#initialise q learner
qlearner = QLearner.QLearner()

#one state as no capacity
qlearner.setNStates((n_lead_time+1,max_cap+1))

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

current_action = [0]*(n_lead_time+1)

for episode in tqdm(range(n_episodes)):
    
    #Random starting state    
    current_capacity = np.random.randint(0,max_cap, size = 7)

    #choose to explore or exploit for each day in the lead time
    for day in range(n_lead_time):
        
        qlearner.setCurrentState((day,current_capacity[day]))

        if np.random.uniform() < explore_prob:
            current_action[day] =  int(np.random.randint(n_prices+1))
        else:
            current_action[day] = qlearner.getBestActionInt((day,current_capacity[day]))

    #set up reward for today
    current_reward = [0]*(n_lead_time+1)

    #set up count for lead time
    lead_time_count = [0]*(n_lead_time+1)

    #set up counter that will track new capacity
    new_capacity = current_capacity.copy()

    #for each customer
    for i in range(n_cust):

        #generate customer threshold
        customer.setThresholdLinear()
        #generate customer lead time
        customer.setLeadTimeUniform(min_lead=1,max_lead=n_lead_time)
        #update lead tiem counter
        lead_time_count[customer.getLeadTime()] += 1
        #assume that willinginess to pay reduces as lead time increases
        customer.setThresholdExact((1 - customer.getLeadTime()/n_lead_time)*customer.getThreshold())

        #if the action is less than the customers willingness to pay
        #and we have space for the customer
        #reward the agent
        if current_action[customer.getLeadTime()] < customer.getThreshold() and new_capacity[customer.getLeadTime()] < max_cap:
            current_reward[customer.getLeadTime()]  += current_action[customer.getLeadTime()] 
            new_capacity[customer.getLeadTime()] +=1
        #else they get nothing

    for day in range(n_lead_time):
         if lead_time_count[day] == 0:
            continue
         else:
            final_reward = current_reward[day]/lead_time_count[day]
            qlearner.setCurrentState((day,current_capacity[day]))
            qlearner.setNewState((day-1,new_capacity[day]))
            qlearner.QLearningUpdate(action = current_action[day], reward = final_reward)

np.save("qtables/cap_lead", qlearner.getQTable())