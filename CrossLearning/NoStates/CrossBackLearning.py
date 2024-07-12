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

reward_crossback = np.zeros((n_episodes,n_macro_reps))

for replication in tqdm(range(n_macro_reps)):

    q_table_crossback = np.zeros((n_prices+1)) 

    for ep in range(n_episodes):

        curr_reward_today = 0

        #choose action at the start of the day
        #Q-learner maximises current estimated value
        #e-Greedy

        #explore
        if np.random.uniform() < explore_prob:
            #random action
            action_price = int(np.random.randint(n_prices+1))
            #action_price = 33
        else:
            #greedy action
            action_price = int(np.argmax(q_table_crossback))

        customer_thresholds = Create_Threshold_Basic_Linear(0,100,n_customers)
        accept_counter = 0

        #go through every customer in the day
        for curr_thresh in customer_thresholds:

            #will customer accept price
            if action_price < curr_thresh:
                
                #if accepted gain reward for revenue
                curr_reward_today += action_price
                accept_counter += 1

            else: 
                curr_reward_today += 0

        
        #update after every day
        #REWARD = Revenue/customer
        final_reward_today = curr_reward_today/n_customers
        reward_crossback[ep,replication]= final_reward_today
        #Q-Learning iteration
        q_table_crossback[action_price] = (1-learn_rate) * q_table_crossback[action_price] + learn_rate * final_reward_today

        percent_accept = accept_counter/n_customers

        for dist in range(1,min(n_prices-action_price, action_price - 0)+1):
            
            #set new learning rate based on distance
            learn_rate_aug = learn_rate/((dist+1)**2)
            
            #cross learn across all actions
            q_table_crossback[action_price+dist] = (1-learn_rate_aug)*q_table_crossback[action_price+dist] + learn_rate_aug*final_reward_today
            #q_table_crossback[action_price-dist] = (1-learn_rate_aug)*q_table_crossback[action_price-dist] + learn_rate_aug*final_reward_today

            #learn backwards dependent on predicted reward
            predicted_reward = percent_accept*(action_price-dist)
            q_table_crossback[action_price-dist] = (1-learn_rate_aug)*q_table_crossback[action_price-dist] + learn_rate_aug*predicted_reward
            
np.save("qtables/CrossBackLearner", q_table_crossback)
np.save("rewards/CrossBackLearnerReward", reward_crossback)