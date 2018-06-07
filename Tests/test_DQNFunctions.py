import unittest
import numpy as np

from RLAgents.Algorithms.DQNFunctions import Q_function, Q_update

class TestDQNFunctions(unittest.TestCase):

    def test_Q_function(self):
        # Scalar
        reward = np.random.rand()
        gamma = np.random.rand()
        Qvalues = np.random.rand(1, 10)
        terminal = np.array(False)
        Q_1 = reward + gamma*np.max(Qvalues)
        self.assertTrue(np.all(Q_1 == Q_function(reward, gamma, Qvalues, terminal)))
        
        # Vector
        reward = [np.random.rand() for i in range(10)]
        gamma = np.random.rand()
        Qvalues = [np.random.rand((10)) for i in range(10)]
        terminal = [np.random.rand() > 0.5 for i in range(10)]
        Q_1 = []
        for r, q, t in zip(reward, Qvalues, terminal):
            Q_1.append(r + gamma*np.max(q)*np.invert(t))
        reward = np.array(reward)
        Qvalues = np.array(Qvalues)
        terminal = np.array(terminal)
        Q_1 = np.array(Q_1)
        self.assertTrue(np.all(Q_1 == Q_function(reward, gamma, Qvalues, terminal)))
        
    def test_Q_update(self):
        # Scalar
        current_Q = np.random.rand(1, 10)
        action = np.random.randint(10)
        reward = np.random.rand()
        gamma = np.random.rand()
        terminal = np.array(False)
        next_Q = np.random.rand(1, 10)
        
        Q_1 = current_Q
        Q_1[0, action] = reward + gamma*np.max(next_Q)
        
        action_vec = np.zeros((1, 10))
        action_vec[0, action] = 1
        
        self.assertTrue(np.all(Q_1 == Q_update(current_Q, action_vec, reward, gamma, next_Q, terminal)))
        
        # Vector
        current_Q = [np.random.rand(1, 10) for _ in range(10)]
        action = [np.random.randint(10) for _ in range(10)]
        reward = [np.random.rand() for _ in range(10)]
        gamma = np.random.rand()
        terminal = [np.random.rand() > 0.5 for i in range(10)]
        next_Q = [np.random.rand(1, 10) for _ in range(10)]
        
        Q_1 = []
        for cQ, a, r, nQ, t in zip(current_Q, action, reward, next_Q, terminal):
            Q = cQ
            if t is True:
                Q[0, action] = r
            else:
                Q[0, action] = r + gamma*np.max(nQ)
            Q_1.append(Q)

                
        action_vec = np.zeros((10, 10))
        action_vec[range(10), action] = 1
        
        current_Q = np.concatenate(current_Q)
        reward = np.array(reward)
        next_Q = np.concatenate(next_Q)
        terminal = np.array(terminal)
        Q_1 = np.concatenate(Q_1)
        
        self.assertTrue(np.all(Q_1 == Q_update(current_Q, action_vec, reward, gamma, next_Q, terminal)))

if __name__ == '__main__':
    unittest.main()