import numpy as np

class QLearner:

    """
    A class to represent the tabular form of reinforcement learning known as Q-learning

    ...

    Attributes
    ----------
    n_states: int
        number of states in the Markov desicion process (MDP) environment

    n_actions: int
        number of actions in the MDP environment

    learn_rate: float
        learning rate/step size of the Q-learning algorithm
        takes value between [0,1]

    discount_factor: float
        discount factor of the Q-learning algorithm
        takes value between [0,1]

    q_table: np.array[float]
        table of size n_states*n_actions that holds all Q-values of the Q-learning algorithm

    Methods
    -------
    getNStates() -> int:
        returns number of states in the MDP environment

    setNStates(number_of_states: int):
        changes Q-Learner's n_states attribute to 'number_of_states'

    getNActions() -> int:
        returns number of actions in the MDP environment

    setNActions(number_of_actions: int):
        changes Q-Learner's n_actions attribute to 'number_of_actions' 
    
    getLearnRate() -> float:
        returns the learning rate/step size of the Q-Learning algorithm

    setLearnRate(learning_rate: float)
        changes the Q-Learner's learn_rate attribute to the float 'learning_rate'
    
    getDiscountFactor() -> float:
        returns the discount factor of the Q-Learning algorithm

    setDiscountFactor(discount_factor: float):
        changes the Q-Learner's discount_factor attribute to the float 'discount_factor'    
    
    getQTable() -> np.array[float]:
        returns the Q-table of the Q-learning algorithm

    setQTable(q_table: np.array[float]):
        changes the Q-Learner's q_table attribute to the np.array[float] 'q_table'
    
    initialiseQTable(q_value: float):
        creates a Q-table of size |States|*|Actions|, with all values initialised at float 'q_table'
    
    getQValue(state: int, action: int) -> float:
        returns the Q-value of a state-action pair

    setQValue(state: int, action: int, q_value: float):
        Changes Q-value of Q(S,A) to the float 'q_value'    

    getBestAction(state: int) -> int:
        returns the actions that corresponds to the argmax Q-table(state,:) 
    """

    def __init__(self, n_states: int = int, n_actions: int = int, learn_rate: float = float, discount_factor: float = float, q_table: np.array[float] = np.array[float]):
        """
        Constructs all the necessary attributes for the QLearner object
        
        Parameters
        ----------
            n_states: int, optional
                number of states in the Markov decision process that the Q-learner interacts with
                (default of int)

            n_actions: int, optional
                number of actions in the Markov decision process that the Q-learner interacts with
                (default is int)

            learn_rate: float, optional
                learning rate/step size of the Q-learning algorithm 
                takes value between [0,1]
                (default is float)

            discount_factor: float, optional
                discount factor of the Q-learning algorithm to help approximate discounted return of state-action pairs
                takes value between [0,1]
                (default is float)
            
            q_table: np.array[float], optional
                Q Table that holds all teh Q-values of each state-action pair
                (default is np.array[float])
        """

        self.__n_states = n_states
        self.__n_actions = n_actions
        self.__learn_rate = learn_rate
        self.__discount_factor = discount_factor
        self.__q_table = q_table

    def getNStates(self) -> int:
        """Returns number of states in the MDP"""
        return self.__n_states
    
    def setNStates(self, number_of_states: int):
        """
        Changes the Q-learner's n_states attribute to int 'number_of_states'

        Parameters
        ----------
            number_of_states: int
                number of states in the MDP
        """
        self.__n_states = number_of_states

    def getNActions(self) -> int:
        """Returns number of actions in the MDP"""
        return self.__n_actions
    
    def setNActions(self, number_of_actions: int):
        """
        Changes the Q-learner's n_actions attribute to int 'number_of_actions'

        Parameters
        ----------
            number_of_actions: int
                number of actions in the MDP
        """
        self.__n_actions = number_of_actions

    def getLearnRate(self) -> float:
        """Returns the learning rate of the Q-learning algorithm"""
        return self.__learn_rate
    
    def setLearnRate(self, learning_rate: float):
        """
        Changes the Q-learner's learn_rate attribute to float 'learning_rate'

        Parameters
        ----------
            learning_rate: float
                learning rate/step size of the Q-learning algorithm
                takes value between [0,1]
        """
        self.__learn_rate = learning_rate

    def getDiscountFactor(self) -> float:
        """Returns the discount factor of the Q-learning algorithm"""
        return self.__discount_factor
    
    def setDiscountFactor(self, discount_factor: float):
        """
        Changes the Q-learner's discount_factor attribute to float 'discount_factor'

        Parameters
        ----------
            discount_factor: float
                discount_factor of the Q-learning algorithm
                takes value between [0,1]
        """
        self.__discount_factor = discount_factor

    def getQTable(self) -> np.array[float]:
        """Returns the Q-table of the Q-learner"""
        return self.__q_table
    
    def setQTable(self, q_table: np.array[float]):
        """
        Changes the q_table attribute to the np.array[float] 'q_table'

        Parameters
        ----------
            q_table: np.array[float]
                Q table used in the Q-learning algorithm, holds the Q-value for every state-action pair 
        """
        self.__q_table = q_table

    def initialiseQTable(self,q_value: float = 0.0):
        """
        Initialises the Q-table for the Q-learning algorithm dependent on the number of states and actions of the MDP

        Parameters
        ----------
            q_value: float, optional
                Initial Q-value that is applied to eery vstate-action pair
                (default is 0)
        """
        #Create an array of size |States|*|Actions|, and then add initial Q-Value
        self.__q_table = np.zeros((self.__n_states,self.__n_actions)) + q_value

    def getQValue(self, state: int, action: int) -> float:
        """
        Returns the Q-value of a specific state-action pair
        
        Parameters
        ----------
            state: int
                state of the wanted value

            action: int
                action of the wanted value
        """
        return self.__q_table[state,action]
    
    def setQValue(self, state: int, action: int, q_value: float):
        """
        Changes a Q-value for a specific state-action pair to teh float 'q_value'

        Parameters
        ----------
            state: int
                state of the value we want to change

            action: int
                action of the value we want to change

            q_value: float
                the float value we want to change the q_value to
        """
    
        self.__q_table[state,action] = q_value

    def getBestAction(self, state: int):
        """
        Returns the action that maximises the Q-values of a specific state
        
        Parameters
        ----------
            state: int
                state to find the argmax of
        """
        return np.argmax(self.__q_table[state,:])
    
    