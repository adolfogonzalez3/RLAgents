
from abc import ABC, abstractmethod
import numpy as np
import random
from itertools import count

from ReplayMemory import SimpleReplay, PrioritizedReplay

from RLAgents.Algorithms.DQNFunctions import Q_function

class BaseDQNet(ABC):

    def __init__(self, epsilon, gamma, replayType='simple', replaySize=1024):
        self.epsilon = epsilon
        self.gamma = gamma
        self.clock = 0
        if replayType == 'simple':
            self.replayMemory = SimpleReplay(replaySize)
        else:
            self.replayMemory = PrioritizedReplay(replaySize)
        
        if type(epsilon) is int or type(epsilon) is float:
            self.epsilon = (epsilon for _ in count())
        
        self.replayType = replayType
    
    def __choose(self, epsilon):
        action = np.zeros(self.current_action.shape)
        if random.random() <= epsilon:
            index = random.randrange(0, action.shape[-1])
        else:
            x = self.Qvalues(self.current_state)
            index = np.argmax(x)
        action[index] = 1
        self.current_action = action
        return self.current_action 
    
    def choose(self, epsilon=None, Training=True):
        if Training is False:
            action = np.zeros(self.current_action.shape)
            x = self.Qvalues(self.current_state, target=True)
            index = np.argmax(x)
            action[index] = 1
            return action
        else:
            if epsilon is None:
                epsilon = next(self.epsilon)
            return self.__choose(epsilon)
        
    def act(self, **kwargs):
        '''Alias for choose.'''
        self.choose(**kwargs)
    
    def choose_random(self):
        return self.__choose(1.0)
    
    def feedback(self, frame, reward, terminal, Training=True):
        #new_state = np.append(frame, self.current_state[...,0:-1], axis=3)
        new_state = self.update_state(frame)
        memory = (self.current_state, self.current_action, reward, new_state, terminal)
        if Training is True:
            if self.replayType == 'prioritized':
                current_Q = np.max(self.Qvalues(self.current_state)*self.current_action)
                T = Q_function(reward, self.gamma, self.Qvalues(new_state, target=True), False)
                self.replayMemory.insert(memory, np.power(current_Q-T, 2))
            else:
                self.replayMemory.insert(memory)
        self.current_state = new_state
        self.terminal = terminal
        
    def batch(self, N):
        batch = self.replayMemory.batch(N)

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