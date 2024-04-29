import numpy as np

class Customer():

    """
    A class to represent a customer.

    ...

    Attributes
    ----------
    threshold: float
        maximum price (£) customer is willing to pay

    lead_time: int
        day(s) between booking and arriving at car park

    LoS: int
        customer's length of stay duration in the car park, measured in days
    

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

    getLoS() -> int:
        Returns a customer's length of stay
    
    setLoSExact(length_of_stay: int):
        Changes the customer's LoS attribute to the value 'length_of_stay'
    
    setLoSUniform(min_LoS: int = 1, max_LoS: int = 2):
        Changes the customer's LoS attribute to a random variable generated from a discrete Uniform[min_LoS,max_LoS] distribution
    """

    def __init__(self, willingness_to_pay_threshold: float = float, lead_time: int = int, length_of_stay: int = int ):
        """
        Constructs all the necessary attributes for the customer object.

        Parameters
        ----------
        willingness_to_pay_threshold: float, optional
            maximum price (£) customer is willing to pay
            converted to 'threshold' attribute (default is float)

        lead_time: int, optional
            days to go from customer booking to arriving at the car park (default is int)

        length_of_stay: int, optional
            duration of customer's stay at the car park measured in days (default is int)
        """
    
        self.__threshold = willingness_to_pay_threshold
        self.__lead_time = lead_time
        self.__LoS = length_of_stay

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
        min_price: int, optional
            minimum price from which customer's willingness to pay threshold will be generated from
            (default is 0)

        max_price: int, optional
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
        lead_time: int
            number of days between customer booking adn then arriving in person
        """
        self.__lead_time = lead_time

    def setLeadTimeUniform(self, min_lead: int = 1, max_lead: int = 7):
        """
        Changes the customer's lead_time attribute to an integer 
        generated from a discrete Uniform[min_lead,max_lead] distribution

        Parameters
        ----------
        min_lead: int, optional
            minumum number of days that can be assigned to a customer as lead time
            (default is 1)

        max_lead: int, optional
            maximum number of days that can be assigned to a customer as lead time
            (default is 7)
        """

        self.__lead_time = np.random.randint(low = min_lead, high = max_lead+1)

    def getLoS(self) -> int:
        """Returns a customer's length of stay."""
        return self.__LoS
    
    def setLoSExact(self, length_of_stay: int):
        """
        Changes a customer's LoS attribute to the value 'length_of_stay'

        Parameters
        ----------
        length_of_stay: int
            number of days the customer is staying in the car park
        """
        self.__LoS = length_of_stay

    def setLoSUniform(self, min_LoS:int = 1, max_LoS:int = 2):
        """
        Changes the customer's LoS attribute to an integer 
        generated from a discrete Uniform[min_LoS,max_LoS] distribution

        Parameters
        ----------
        min_LoS: int, optional
            minumum number of days that can be assigned to a customer's length of stay
            (default is 1)

        max_LoS: int, optional
            maximum number of days that can be assigned to a customer's length of stay
            (default is 2)
        """

        self.__LoS = np.random.randint(low = min_LoS, high = max_LoS+1)