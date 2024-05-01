import numpy as np

class Customer():

    """
    A class to represent a customer.

    ...

    Attributes
    ----------
    threshold: list[float] 
        maximum price (£) customer is willing to pay for each product

    lead_time: int
        day(s) between booking and arriving at car park

    LoS: int
        customer's length of stay duration in the car park, measured in days

    segment: str
        shows which segmentation the customer lies in
        possible options are "business", "leisure" or "switch"
    
    buy_type: str
        describes how a customer chooses to buy a product if they are willing to buy multiple
        possble options are "min_buy", "max_buy", "rank_buy", "utility"

    preference: list[int]
        ordered list of which product customer wants to buy. most preferred to least

    Methods
    -------
    getThreshold() -> list[float] :
        Returns willingness to pay threshold of customer for each product

    setThresholdExact(threshold: list[float]):
        Changes the customer's threshold attribute to the list of floats 'threshold'      
    
    setThresholdLinear(min_price: int = 0, max_price: int = 100, n_products: int = 1, sort: bool = True):
        Changes the customer's threshold attribute to list of random variables generated from a linearly decreasing probability mass function

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

    getSegment() -> str:
        Returns a customer's segmentation

    setSegmentExact(segment: str):
        Changes a customer's segment attribute to the string 'segment'

    setSegmentUniform():
        Changes a customer's segment attribute uniformly randomly to 1 of the 3 valid options

    getBuyType() -> str:
        Return's a customer's buy type

    setBuyTypeExact(buy_type: str):
        Changes a customer's buy_type attribute to the string 'buy_type'

    setBuyTypeUniform():
        Changes a customer's buy_type attribute unifromly randomly to 1 of the 4 valid options
    
    getPreference() -> list[int]:
        Returns an ordered list of preferred products for the customer

    setPreferenceExact(order: list[int]):
        Changes a customer's preference attribute to the list[int] 'order'
    
    setPreferenceRandom(n_products: int = 2):
        Changes a customer's preference attribute to a random list[int], containing each product index once

    setPreferenceFromType(offered_prices: list[float]):
        Changes a customer's preference attribute according to their buy_type attribute

    """

    def __init__(self,
                willingness_to_pay_threshold: list[float] = list[float],
                lead_time: int = int,
                length_of_stay: int = int, 
                segment: str = str, 
                buy_type: str = str, 
                preference: list[int] = list[int]):
        """
        Constructs all the necessary attributes for the customer object.

        Parameters
        ----------
            willingness_to_pay_threshold: list[float], optional
                maximum price (£) customer is willing to pay for each product
                converted to 'threshold' attribute (default is list[float])

            lead_time: int, optional
                days to go from customer booking to arriving at the car park (default is int)

            length_of_stay: int, optional
                duration of customer's stay at the car park measured in days (default is int)

            segment: str, optional
                which segmentation the customer falls under (default is str)

            buy_type: str, optional
                describes how the customer chooses between different prices (default is str)

            preference: list[int], optional
                ordered list of the preference of products the customer wants to buy

        """
    
        self.__threshold = willingness_to_pay_threshold
        self.__lead_time = lead_time
        self.__LoS = length_of_stay
        self.__segment = segment
        self.__buy_type = buy_type
        self.__preference = preference

    def getThreshold(self) -> list[float] :
        """ Returns willingness to pay threshold of customer. """
        #if we only have one product, just return threshold as float rather than list 
        if len(self.__threshold) == 1:
            return float(self.__threshold[0])
        else:
            return self.__threshold  
    
    def setThresholdExact(self, threshold: list[float]):
        """ 
        Changes the customer's threshold attribute to the value 'threshold'.
        
        Parameters
        ----------
            threshold : list[float] 
                maximum price (£) customer is willing to pay for each product
        """
        self.__threshold = threshold

    def setThresholdLinear(self, min_price: int = 0, max_price: int = 100, n_products: int = 1, sort: bool = True):
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

            n_products: int, optional
                number of different (tiered) products we wish to create thresholds for
                (default is 1)

            sort: bool, optional
                defines if you wish the random thresholds to be sorted from highest to lowest, or to stay in given random order
                (default is True)
            
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
                                     p = probabilities,
                                     size = n_products)
        
        #assign random threshold to attribute
        #either as sorted or unsorted list of floats depending on sort: bool operator
        self.__threshold = list(map(float,threshold))*(sort == False) + (sort == True)*sorted(list(map(float,threshold)),reverse = sort)

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

    def getSegment(self) -> str:
        """Returns a customer's segmentation"""
        return self.__segment
    
    def setSegmentExact(self, segment: str):
        """
        Changes a customer's segment attribute to the string 'segment'

        Parameters
        ----------
            segment: str
                segmentation customer falls under ("business", "leisure", "switch")
        """
        self.__segment = segment

    def setSegmentUniform(self):
        """
        Changes a customer's segment attribute to either "business", "leisure" or "switch" uniformly randomly
        """
        self.__segment = np.random.choice(a = ["business", "leisure", "switch"], p = [1/3,1/3,1/3])

    def getBuyType(self) -> str:
        """Returns a customer's buy type - indicating how they choose a product if they are willing to buy multiple"""
        return self.__buy_type
    
    def setBuyTypeExact(self, buy_type: str):
        """
        Changes a customer's buy_type attribute to the string 'buy_type'

        Parameters
        ----------
            buy_type: str
                defines how customer buys if given choice of multiple products they wish to buy
        """
        self.__buy_type = buy_type

    def setBuyTypeUniform(self):
        """
        Changes a customer's buy_type attribute to either "min_buy", "max_buy", "rank_buy", "utility"
        """
        self.__buy_type = np.random.choice(a = ["min_buy", "max_buy", "rank_buy", "utility"], p = [1/4,1/4,1/4,1/4])

    def getPreference(self) -> list[int]:
        """Returns an ordered list of preferred products for the customer"""
        return self.__preference
    
    def setPreferenceExact(self, order: list[int]):
        """
        Changes a customer's preference attribute to the list[int] 'order'.

        Parameters
        ----------
            order: list[int]
                ordered list of ranked preference, from most preferred to least
        """
        self.__preference = order

    def setPreferenceRandom(self, n_products: int = 2):
        """
        Changes a customer's preference attribute to a random list[int] of the half open set of integers [0,..,'n_products')

        Parameters
        ----------
            n_products: int, optional
                number of products to be ranked (default is 2)
        """
        self.__preference = list(np.random.choice(n_products, size = n_products, replace = False))

    def setPreferenceFromType(self, offered_prices: list[float]):
        """
        Changes a customer's preference attribute to a list[int] according to their buy_type attribute
        The offered prices are needed as an inupt in order to rank the products for most customer types

        Parameters
        ----------
            offered_prices: list[float]
                list of prices that are offered, used in order to create preference list
            
        """
        #find how many products we need to order
        n_products = len(offered_prices)

        #create a  new list that we can change without affecting the original
        aug_prices = offered_prices.copy()

        if self.__buy_type == "min_buy":

            #if customer is min buying we need to order product indicies from smallest to largest

            #produce a list of products in decsending order
            #the list will be the indicies of the product
            
            self.__preference = list(np.argsort(aug_prices))

            #if you dont trust numpy use the code I wrote before I found the np.argsort() function

            # while len(pref_order) != n_products:
            #     #while all products are not ordered

            #     #find position of current minimum
            #     current_min_pos = aug_prices.index(min(aug_prices))

            #     #append this to the preference list
            #     pref_order.append(current_min_pos)

            #     #set current minimum to infinity so next minimum can be found
            #     aug_prices[current_min_pos] = float('inf')                

            # #once all products are accounted for,
            # #set customer's preference attribute to pref_order
            # self.__preference = pref_order
        
        elif self.__buy_type == "max_buy":
            #if customer is max buying we need to order product indicies from largest to smallest

            #produce a list of products in ascending order
            #the list will be the indicies of the product

            #make a negative version of the list
            neg_prices = [-price for price in aug_prices]

            self.__preference = list(np.argsort(neg_prices))
        
        elif self.__buy_type == "rank_buy":
            
            # assuming that each rank buy customer has their own independent random ranking
            self.setPreferenceRandom(n_products = n_products)

        elif self.__buy_type == "utility":
            
            
            #list giving utility for customer for each product
            #we have found negative utility so that we can use np.argsort()
            neg_utility_list = [price - threshold for price, threshold in zip(aug_prices,self.__threshold)]

            self.__preference = list(np.argsort(neg_utility_list))