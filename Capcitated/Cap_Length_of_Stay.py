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
explore_prob = 1

#maximum length of stay a customer can stay for
n_length_of_stay = 2

#maximum capacity fo the car park
max_cap = (50,50)
max_cap_with_empty = (max_cap[0]+1,max_cap[1]+1)

#initialise q learner
qlearner = QLearner.QLearner()

#one state as no capacity
qlearner.setNStates(max_cap_with_empty)

#number of prices we can set as actions
qlearner.setNActions((n_prices+1,n_prices+1))

#learning rate/step size of algorithm
qlearner.setLearnRate(0.1)

#discount factor = 0 as no definition of new state
qlearner.setDiscountFactor(.1)

#initialise q table
#qlearner.initialiseQTable()

#load in older q-table to update
qlearner.setQTable(np.load("qtables/cap_LoS.npy"))

#initialise a cutomer
customer = Customers.Customer()

current_action = [0]*n_length_of_stay

current_action = [0]*n_length_of_stay

for episode in tqdm(range(n_episodes)):

    current_capacity = np.random.randint(0,max_cap, size = 2)

    qlearner.setCurrentState((current_capacity[0],current_capacity[1]))

    if np.random.uniform() < explore_prob:
        current_action =  tuple(np.random.randint(n_prices+1,size = n_length_of_stay))
    else:
        current_action = qlearner.getBestAction((current_capacity[0],current_capacity[1]))

    #set up reward for today
    current_reward = 0

    #set up vector to store new capacity as customers book
    new_capacity = current_capacity.copy()

    #for each customer
    for i in range(n_cust):
        #generate customer threshold
        customer.setThresholdLinear()
        #generate customer length of stay
        customer.setLoSUniform(min_LoS=1,max_LoS=2)
        #calculate total threshold dependent on LoS and price per day willingness to pay
        customer.setThresholdExact(customer.getLoS()*customer.getThreshold())

        #if the action is less than the customers willingness to pay
        #and we have space for all days the customer is booking over
        #reward the agent
        #print(current_capacity,new_capacity,max_cap)
        if sum(current_action[:customer.getLoS()]) < customer.getThreshold() and all(np.array(new_capacity[:customer.getLoS()]) < np.array(max_cap[:customer.getLoS()])):
            current_reward += sum(current_action[:customer.getLoS()])
            
            #update capacity
            for i in range(customer.getLoS()):
                new_capacity[i] += 1
        #else they get nothing
   
    final_reward = current_reward/n_cust

    qlearner.setNewState((new_capacity[0],new_capacity[1]))

    qlearner.QLearningUpdate(action = current_action, reward = final_reward)


np.save("qtables/cap_LoS", qlearner.getQTable())