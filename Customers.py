import numpy as np

class Customer():

    """
    A class to represent a customer.

    ...

    Attributes
    ----------
    threshold : float
        maximum price (£) customer is willing to pay

    lead_time : float
        day(s) between booking and arriving at car park
    

    Methods
    -------
    getThreshold() -> float:
        Returns willingness to pay threshold of customer

    setThresholdExact(threshold: float):
        Changes the customer's threshold attribute to the value 'threshold'      
    
    setThresholdLinear(min_price: int = 0, max_price: int = 100):
        Changes the customer's threshold attribute to a random variable generated from a linearly decreasing probability mass function

    getLeadTime() -> int:
        Returns customer's lead time
    
    setLeadTimeExact(lead_time: int):
        Changes the customer's lead_time attribute to the value 'lead_time'

    set LeadTimeUniform(min_lead: int = 1, max_lead: int = 7):
        Changes the customer's lead_time attribute to a random variable generated from a disctrete Uniform[min_lead,max_lead] distribution
    """

    def __init__(self, willingness_to_pay_threshold: float = float, lead_time: int = int):
        """
        Constructs all the necessary attributes for the customer object.

        Parameters
        ----------
        willingness_to_pay_threshold : float, optional
            maximum price (£) customer is willing to pay
            converted to 'threshold' attribute (default is float)

        lead_time : int, optional
            days to go from customer booking to arriving at the car park (default is int)
        """
    
        self.__threshold = willingness_to_pay_threshold
        self.__lead_time = lead_time



    def getThreshold(self) -> float:
        """ Returns willingness to pay threshold of customer. """
        return self.__threshold  
    


    def setThresholdExact(self, threshold: float):
        """ 
        Changes the customer's threshold attribute to the value 'threshold'.
        
        Parameters
        ----------
        threshold : float
            maximum price (£) customer is willing to pay
        """
        self.__threshold = threshold



    def setThresholdLinear(self, min_price: int = 0, max_price: int = 100):
        """
        Changes the customer's threshold attribute 
        according to a linearly decreasing probability mass function

        Parameters
        ----------
        min_price : int, optional
            minimum price from which customer's willingness to pay threshold will be generated from
            (default is 0)

        max_price : int,optional
            maximum price from which customer's willingness to pay threshold will be generated from
            (default is 100)
        """
        #create range from minumum and maximum parameters
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


    def getLeadTime(self) -> int:
        """ Returns customer's lead time. """
        return self.__lead_time

    def setLeadTimeExact(self, lead_time: int):
        """ 
        Changes the customer's lead_time attribute to the value 'lead_time'.
        
        Parameters
        ----------
        lead_time : int
            number of days between customer booking adn then arriving in person
        """
        self.__lead_time = lead_time

    def setLeadTimeUniform(self, min_lead: int = 1, max_lead: int = 7):
        """
        Changes the customer's lead_time attribute to a integer 
        generated from a discrete Uniform[min_lead,max_lead] distribution

        Parameters
        ----------
        min_lead : int
            minumum number of days that can be assigned to a customer as lead time
            (default is 1)

        max_lead : int 
            maximum number of days that can be assigned to a customer as lead time
            (default is 7)
        """

        self.__lead_time = np.random.randint(low = min_lead, high = max_lead+1)