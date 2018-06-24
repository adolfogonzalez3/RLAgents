
import numpy as np


def Q_function(reward, gamma, Qvalues, terminal):
    return reward + gamma*np.max(Qvalues, axis=1)*np.invert(terminal)
    
def Q_update(current_Q, action_performed, reward, gamma, next_Q, terminal):
    current_Q[action_performed==1] = Q_function(reward, gamma, next_Q, terminal)
    return current_Q
    
def batch(self, memory, N):
    batch = memory.batch(N)

    pseq_batch = np.concatenate([b[0] for b in batch], axis=0)
    action_batch = np.stack([b[1] for b in batch])
    reward_batch = np.array([b[2] for b in batch])
    seq_batch = np.concatenate([b[3] for b in batch], axis=0)
    term_batch = np.array([b[4] for b in batch])
    
    return pseq_batch, action_batch, reward_batch, seq_batch, term_batch