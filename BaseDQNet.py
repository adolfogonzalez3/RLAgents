
from abc import ABC, abstractmethod
import numpy as np
import random

from itertools import count

from ReplayMemory import SimpleReplay, PrioritizedReplay

from RLAgents.Algorithms.DQNFunctions import Q_function

class BaseDQNet(ABC):

    def __init__(self, epsilon, gamma, K, replayType='simple', replaySize=1024):
        self.epsilon = epsilon
        self.gamma = gamma
        self.K = K
        self.clock = 0
        if replayType == 'simple':
            self.replayMemory = SimpleReplay(replaySize)
        else:
            self.replayMemory = PrioritizedReplay(replaySize)
        
        if type(epsilon) is int or type(epsilon) is float:
            self.epsilon = (epsilon for _ in count())
        
        self.replayType = replayType
        
    def initialize(self, current_state, current_action):
        self.current_state = current_state
        self.current_action = current_action
        self.terminal = False 
        
    def choose(self, Training=True):
        if Training is False:
            action = np.zeros(self.current_action.shape)
            x = self.Qvalues(self.current_state, target=True)
            index = np.argmax(x)
            action[index] = 1
            return action
        else:
            e = next(self.epsilon)
            if self.clock%self.K == 0:
                return self.__choose(e)
            else:
                return self.current_action
        
    def __choose(self, epsilon):
        action = np.zeros(self.current_action.shape)
        if random.random() <= epsilon:
            index = random.randint(0, len(action.shape))
        else:
            x = self.Qvalues(self.current_state)
            index = np.argmax(x)
        action[index] = 1
        self.current_action = action
        return self.current_action
    
    def feedback(self, frame, reward, terminal, Training=True):
        new_state = np.append(frame, self.current_state[...,0:-1], axis=3)
        memory = (self.current_state, self.current_action, reward, new_state, terminal)
        if Training is True:
            if self.replayType == 'prioritized':
                current_Q = np.max(self.Qvalues(self.current_state)*self.current_action)
                T = Q_function(reward, self.gamma, self.Qvalues(new_state, target=True), False)
                self.replayMemory.insert(memory, abs(current_Q-T))
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
    def Qvalues(self, state, target=False):
        pass

    @abstractmethod
    def train(self):
        pass
        
    @abstractmethod
    def save(self):
        pass