import numpy as np

"""
MECHANICAL ENGINEERING POST-GRADUATE PROGRAM
UNIVERSIDADE FEDERAL DO ABC - SANTO ANDRÉ, BRASIL

NOME: RAFAEL COSTA FERNANDES
RA: 21201920754
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIPTION:
    Generates a T iterations delayed neural network input.
"""

class dl_in_gen():
    
    def __init__(self, T, state_size, action_size):
        self.hist_size = state_size+action_size+1
        self.deep_learning_in_size = self.hist_size*T
        self.reset()
        
    def reset(self):
        self.deep_learning_input = np.zeros(self.deep_learning_in_size)
        
    def dl_input(self, states, actions):
        
        for state, action in zip(states, actions):
            state_t = np.concatenate((action, state))
            self.deep_learning_input = np.roll(self.deep_learning_input, -self.hist_size)
            self.deep_learning_input[-self.hist_size:] = state_t
        return self.deep_learning_input