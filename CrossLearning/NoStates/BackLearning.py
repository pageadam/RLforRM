import numpy as np
from tqdm import tqdm

#Function for creating customer thresholds
def Create_Threshold_Basic_Linear(min_price,max_price,amount_of_thresholds):
    y = [max_price - i for i in range(min_price,max_price+1)]
    thresh = np.random.choice(a = range(min_price,max_price+1),
                               p = [i/sum(y) for i in y],
                                 size = amount_of_thresholds)
    return list(thresh)

#step size/ learning rate/ alpha
learn_rate = 0.1

#number of episodes
n_episodes = 10**5

#number of macro replications
n_macro_reps = 30

#exploration probability for epsilon greedy algorithm
explore_prob = .1

#number of customers
n_customers = 100

#number of prices
n_prices = 100

#set seed
np.random.seed(11)

#holder for reward
reward_back = np.zeros((n_episodes,n_macro_reps))

#for each macro replication
for replication in tqdm(range(n_macro_reps)):

    #initialise empty q-table
    q_table_backend = np.zeros((n_prices+1)) 

    #for eacch episode
    for ep in range(n_episodes):

        #initilise reward
        curr_reward_today = 0

        #choose action at the start of the day
        #Q-learner maximises current estimated value
        #e-Greedy

        #explore
        if np.random.uniform() < explore_prob:
            #random action
            action_price = int(np.random.randint(n_prices+1))
        else:
            #greedy action
            action_price = int(np.argmax(q_table_backend))

        customer_thresholds = Create_Threshold_Basic_Linear(0,100,n_customers)

        accept_counter = 0

        #go through every customer in the day
        for curr_thresh in customer_thresholds:

            #will customer accept price
            if action_price < curr_thresh:
                
                #if accepted gain reward for revenue
                curr_reward_today += action_price
                accept_counter += 1


        
        #update after every day
                
        #REWARD = Revenue/customer
        final_reward_today = curr_reward_today/n_customers
        #log reward
        reward_back[ep,replication] = final_reward_today
        #Q-Learning iteration
        q_table_backend[action_price] = (1-learn_rate) * q_table_backend[action_price] + learn_rate * final_reward_today


        #calculate the percetnage of acceptances
        percent_accept = accept_counter/n_customers

        #for each price up to the current price
        for dist in range(action_price):

            #augment the learning rate dependent on distance
            learn_rate_aug = learn_rate/((dist+1)**2)

            #estimate the predictive reward based on current observation
            predicted_reward = percent_accept*(action_price-dist)

            #back-q-learner-iteration
            q_table_backend[action_price-dist] = (1-learn_rate_aug)*q_table_backend[action_price-dist] + learn_rate_aug*predicted_reward

np.save("qtables/BackLearner", q_table_backend)
np.save("rewards/BackLearnerReward", reward_back)