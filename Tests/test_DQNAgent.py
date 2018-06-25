import unittest
import numpy as np

from RLAgents import DQNAgent

from create_test_model import create_test_model

class TestDQNAgent(unittest.TestCase):

    def test_initialize(self):
        '''Tests the shape of the choice from the choose_random method.'''
        model_lambda = lambda: create_test_model((10, 4), 2)
        agent = DQNAgent(model_lambda)
        self.assertTrue(agent.initialize(np.zeros((10, )), np.zeros(2)))

    def test_choose_random_shape(self):
        '''Tests the shape of the choice from the choose_random method.'''
        model_lambda = lambda: create_test_model((10, 4), 2)
        agent = DQNAgent(model_lambda)
        agent.initialize(np.zeros((10, )), np.zeros(2))
        self.assertEqual(agent.choose_random().shape, (2,))
    
    def test_choose_shape(self):
        '''Tests the shape of the choice from the choose_random method.'''
        model_lambda = lambda: create_test_model((10, 4), 2)
        agent = DQNAgent(model_lambda)
        agent.initialize(np.zeros((10, )), np.zeros(2))
        self.assertEqual(agent.choose().shape, (2,))
        
    def test_save(self):
        model_lambda = lambda: create_test_model((10, 4), 2)
        agent = DQNAgent(model_lambda)
        agent.initialize(np.zeros((10, )), np.zeros(2))
        agent.save('TEST')

if __name__ == '__main__':
    unittest.main()