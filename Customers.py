import numpy as np

#some set up variables for the create_thresholh() function

# minimum price customer is willing to accept
min_price = 0 

#maximum price customer is willing to accept
max_price = 100 

#vector that reverses range
y = [max_price - i for i in range(min_price,max_price+1)]

# returns a price thrshold from a linearly decreasing pmf
def create_threshold():

    #creates linearly decreasing probability of picking threshold
    probs = [i/sum(y) for i in y]

    #chooses the random threshold
    thresh = np.random.choice(a = range(min_price,max_price+1),
                              p = probs)
    
    return thresh
