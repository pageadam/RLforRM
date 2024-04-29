import numpy as np

class Customer():

    """
    A class to represent a customer.

    ...

    Attributes
    ----------
    threshold : float
        maximum price (£) customer is willing to pay 
    

    Methods
    -------
    getThreshold():
        Returns willingness to pay threshold of customer

    setThreshold(threshold:float):
        Changes the customer's threshold attribute to the value 'threshold'      
    """

    def __init__(self, willingness_to_pay_threshold:float = float ):

        """
        Constructs all the necessary attributes for the customer object.

        Parameters
        ----------
        willingness_to_pay_threshold : float, optional
            maximum price (£) customer is willing to pay
            converted to 'threshold' attribute (default is float)
        """
    
        self.__threshold = willingness_to_pay_threshold



    def getThreshold(self) -> float:
        """ Returns willingness to pay threshold of customer."""
        return self.__threshold
    


    def setThresholdExact(self, threshold:float):
        """ 
        Changes the customer's threshold attribute to the value 'threshold'.
        
        Parameters
        ----------
        threshold : float
            maximum price (£) customer is willing to pay
        """
        self.__threshold = threshold

    def setThresholdLinear(self,min_price:float = 0, max_price:float = 100):
        """
        Changes the customer's threshold attribute 
        according to a linearly decreasing probability mass function

        Parameters
        ----------
        min_price : float, optional
            minimum price from which customer's willingness to pay threshold will be generated from
            (default is 0)

        max_price : float,optional
            maximum price from which customer's willingness to pay threshold will be generated from
            (default is 100)
        """
        #create range from minumu and maximum parameters
        price_range = range(min_price,max_price+1)

        #create sum now to avoid repeating function
        total = sum(price_range)

        #creates linearly decreasing probability of picking threshold
        #large amounts of customers have small thresholds
        #few people have large thresholds 
        probabilities = [price/total for price in reversed(price_range)]

        #chooses the random threshold based on linearly decreasing probabilities
        threshold = np.random.choice(a = price_range,
                                     p = probabilities)
        
        #assign random threshold to attribute
        self.__threshold = float(threshold)

    
