
import numpy as np


def Q_function(reward, gamma, Qvalues, terminal):
    return reward + gamma*np.max(Qvalues, axis=1)*np.invert(terminal)
    
def Q_update(current_Q, action_performed, reward, gamma, next_Q, terminal):
    current_Q[action_performed==1] = Q_function(reward, gamma, next_Q, terminal)
    return current_Q