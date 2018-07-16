
from abc import ABC, abstractmethod
import numpy as np
import random
from itertools import count

from ReplayMemory import SimpleReplay, PrioritizedReplay

from RLAgents.Algorithms.DQNFunctions import Q_function

class BaseDQNet(ABC):
    '''A Base class which all agent subclasses must inherit.
    
    The actions described in this base class are expected methods of all agent
    subclasses. There exists within this base class methods which are implemented.
    These implemented methods rely on the abstract methods which are expected to be
    implemented by any agent subclass which inherits this class.
    
    Currently this base class is written with implementing Deep Q algorithms,
    however in the future a base class should be made which describes the very
    basic of the methods which an agent may be expected to perform.
    
    '''

    def __init__(self, epsilon, gamma, replay_type='simple', replay_size=1024):
        '''Create an agent that must have an epsilon, gamma, replay_type, and replay_size.'''
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Currently defined replay types are a simple replay and a 
        # prioritized replay.
        if replay_type == 'simple':
            self.replay_memory = SimpleReplay(replay_size)
        else:
            self.replay_memory = PrioritizedReplay(replay_size)
        
        if type(epsilon) is int or type(epsilon) is float:
            self.epsilon = (epsilon for _ in count())
        
        self.replay_type = replay_type
    
    def __greedy_epsilon(self, epsilon):
        '''Choose an action with policy greedy-epsilon.'''
        # Rely on current stored action for shape
        action = np.zeros(self.current_action.shape)
        if random.random() <= epsilon:
            # Choose an action randomly
            # Dimensions of action.shape should be: (1, number_of_actions)
            index = random.randrange(0, action.shape[-1])
        else:
            x = self.Qvalues(self.current_state)
            index = np.argmax(x)
        action[index] = 1
        self.current_action = action
        return self.current_action 
    
    def choose(self):
        '''Choose an action based off greedy-epsilon policy.'''
        epsilon = next(self.epsilon)
        return self.__greedy_epsilon(epsilon)
        
    def act(self):
        '''Alias for choose.'''
        self.choose()
    
    def choose_random(self):
        '''Randomly choose an action.'''
        return self.__greedy_epsilon(1.0)
    
    def feedback(self, frame, reward, terminal, Training=True):
        '''Receive feedback from the environment.
        
        If Training is True then the memory instance is inserted into the
        replay memory otherwise the memory is discarded.
        '''
        new_state = self.update_state(frame)
        
        if Training is True:
            memory = (self.current_state, self.current_action, reward, new_state, terminal)
            if self.replay_type == 'prioritized':
                # Prioritized memory requires the error value along with the 
                # memory instance therefore the error is produced by getting the 
                # anticipated Q values for the current state and comparing them to
                # the true values for current state.
                # The network produces the anticipated Q values for a state.
                # The Q function produces the true values for the current state.
                current_Q = np.max(self.Qvalues(self.current_state)*self.current_action)
                T = Q_function(reward, self.gamma, self.Qvalues(new_state, target=True), False)
                self.replay_memory.insert(memory, np.power(current_Q-T, 2))
            else:
                self.replay_memory.insert(memory)
        self.current_state = new_state
        self.terminal = terminal
        
    def batch(self, N):
        '''Produce a batch of instances from the replay memory.'''
        batch = self.replay_memory.batch(N)

        pseq_batch = np.concatenate([b[0] for b in batch], axis=0)
        action_batch = np.stack([b[1] for b in batch])
        reward_batch = np.array([b[2] for b in batch])
        seq_batch = np.concatenate([b[3] for b in batch], axis=0)
        term_batch = np.array([b[4] for b in batch])
        
        return pseq_batch, action_batch, reward_batch, seq_batch, term_batch
    
    @abstractmethod
    def initialize(self, state, action):
        pass
    
    @abstractmethod
    def Qvalues(self, state, target=False):
        pass

    @abstractmethod
    def train(self):
        pass
        
    @abstractmethod
    def save(self, name):
        pass
        
    @abstractmethod    
    def update_state(self, state):
        '''Create a new state given input.'''
        pass