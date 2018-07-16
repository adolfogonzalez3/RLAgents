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

    def __init__(self, model, replay_size=1024, batch_size=32, gamma=0.99,
                epsilon=0.1, tau=0.0001, name='Agent', replay_type='simple',
                with_target=True, previous_frames=4):
        '''Create Agent from model description file.'''
        super().__init__(epsilon, gamma, replayType, memory_size)
        self.tau = tau
        self.batch_size = batch_size
        self.with_target = with_target
        self.previous_frames = previous_frames
        
        self.model = FFNetAsync(model)
        # If using a target network, create a duplicate network from the same
        # network architecture file
        if with_target is True:
            self.target_model = FFNetAsync(model)

    def save(self, name):
        '''Save the networks using the keras save function.'''
        self.model.save('{}.h5'.format(name))
        if self.with_target is True:
            self.target_model.save('{}_target.h5'.format(name))

    def train(self):
        '''Train the Agent.'''
        prev_state, actions, rewards, next_state, terms = self.batch(self.batch_size)
        
        # If not using a target network, use the online network instead.
        if self.with_target is True:
            target_model = self.target_model
        else:
            target_model = self.model
        
        # Handles the predictions asynchronously
        target_model.predict_on_batch(next_state)
        self.model.predict_on_batch(prev_state)
        
        next_Qvalues = target_model.collect()
        current_Qvalues = self.model.collect()
        
        # Updates the Q value based on current state given the actions
        # Q_online(s_t, a) = r + gamma*Q_target(s_t+1, a)
        # or if terminal is True
        # Q_online(s_t, a) = r
        new_Qvalues = Q_update(current_Qvalues, actions, rewards, self.gamma, next_Qvalues, terms)
        
        self.model.train_on_batch(prev_state, new_Qvalues)
        
        # If using a target network, update the weights of the target using the
        # online network's weights.
        if self.with_target is True:
            self.update_weights()

    def update_weights(self):
        '''Update the target network using the online network's weights.'''
        self.model.get_weights()
        self.target_model.get_weights()
    
        weights = self.model.collect()
        target_weights = self.target_model.collect()
        for i in range(len(target_weights)):
            target_weights[i] = target_weights[i]*(1-self.tau) + weights[i]*self.tau
        self.target_model.set_weights(target_weights)
        
    def Qvalues(self, state, target=False):
        '''Return the Q values for the given state.
        
        If target is True then use the target network to predict the Q values.
        '''
        if target is True and self.with_target is True:
            model = self.target_model
        else:
            model = self.model
        model.predict_on_batch(state)
        x = model.collect()
        return x
        
    def update_state(self, state):
        '''Create a new state given input.
        
        First reshapes the state so that it may appended.
        
        The shape is (1, width, height, number of states).
        
        Currently casting the new state as float 16 to save space while maintaining
        compatibility with different feature formats ( catergorical, RGB ).
        '''
        reshaped_state = state.reshape((1,) + state.shape + (1,))
        return np.append(reshaped_state, self.current_state[...,0:-1], axis=-1).astype(np.int8)
        
    def initialize(self, state, action):
        '''Initialize the agent with state and action.'''
        
        # Stacks the agents along a new axis.
        stacked_states = [state for _ in range(self.previous_frames)]
        new_state = np.stack(stacked_states, axis=-1)
        
        # Reshaped state shape is (1, width, height, number of prior states)
        single_state = new_state.reshape((1,) + new_state.shape)
        self.current_state = single_state
        self.current_action = action
        self.terminal = False 
        return True