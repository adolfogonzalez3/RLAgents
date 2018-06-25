import numpy as np
import random

from RLAgents.Algorithms.DQNFunctions import Q_function, Q_update 
from RLAgents.BaseDQNet import BaseDQNet

from NNets import FFNetAsync, FFNetEnsemble
        
class DQNAgent(BaseDQNet):
    '''An agent class used in testing reinforcement learning algorithms.

    This class is made with the purpose that it would allow multiple agents to
    be trained concurrently in a single game so the majority of their
    work should be hidden behind this class.
    '''

    def __init__(self, model, memory_size=1024,
                batch_size=32, gamma=0.99, epsilon=0.1,
                name='Agent', replayType='simple', with_target=True):
        '''Create Agent from model description file.'''
        super().__init__(epsilon, gamma, replayType, memory_size)
        self.batch_size = batch_size
        self.model = FFNetAsync(model)
        self.with_target = with_target
        if with_target is True:
            self.target_model = FFNetAsync(model)

    def save(self, name):
        self.model.save('{}.h5'.format(name))
        if self.with_target is True:
            self.target_model.save('{}_target.h5'.format(name))

    def train(self):
        '''Train the Agent.'''
        pseq, actions, rewards, seq, terms = self.batch(self.batch_size)
        
        if self.with_target is True:
            target_model = self.target_model
        else:
            target_model = self.model
        
        target_model.predict_on_batch(seq)
        self.model.predict_on_batch(pseq)
        
        nextQ = target_model.collect()
        currentQ = self.model.collect()
        
        newQ = Q_update(currentQ, actions, rewards, self.gamma, nextQ, terms)
        
        self.model.train_on_batch(pseq, newQ)
        
        if self.with_target is True:
            self.update_weights()

    def update_weights(self):
        self.model.get_weights()
        self.target_model.get_weights()
    
        weights = self.model.collect()
        target_weights = self.target_model.collect()
        for i in range(len(target_weights)):
            target_weights[i] = target_weights[i]*(0.99) + weights[i]*0.01
        self.target_model.set_weights(target_weights)
            
    def get_epsilon(self):
        return next(self.epsilon)
        
    def Qvalues(self, state, target=False):
        if target is True and self.with_target is True:
            model = self.target_model
        else:
            model = self.model
        model.predict_on_batch(state)
        x = model.collect()
        return x
        
    def update_state(self, state):
        '''Create a new state given input.'''
        reshaped_state = state.reshape((1,) + state.shape + (1,))
        return np.append(reshaped_state, self.current_state[...,0:-1], axis=3)
        
    def initialize(self, state, action):
        new_state = np.stack((state, state, state, state), axis=-1)
        single_state = new_state.reshape((1,) + new_state.shape)
        self.current_state = single_state
        self.current_action = action
        self.terminal = False 
        return True