import sys
from quadrotor_env import quad, render, animation
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from quadrotor_env import quad, render, animation
from model import ActorCritic

"""
MECHANICAL ENGINEERING POST-GRADUATE PROGRAM
UNIVERSIDADE FEDERAL DO ABC - SANTO ANDRÉ, BRASIL

NOME: RAFAEL COSTA FERNANDES
RA: 21201920754
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIPTION:
    PPO testing algorithm (no training, only forward passes)
"""

time_int_step = 0.01
max_timesteps = 1000
T = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = quad(time_int_step, max_timesteps, euler=0, direct_control=1, deep_learning=1, T=T, debug=0)
state_dim = env.deep_learning_in_size
policy = ActorCritic(state_dim, action_dim=4, action_std=0).to(device)


#LOAD TRAINED POLICY
try:
    policy.load_state_dict(torch.load('PPO_continuous_solved_drone.pth',map_location=device))
    print('Saved policy loaded')
except:
    print('Could not load policy')
    sys.exit(1)

#PLOTTER SETUP
print_states = [0, 2, 4, 6, 7, 8, 9, 10, 11, 12]
plot_labels = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'f1', 'f2', 'f3', 'f4']
line_styles = ['-', '-', '-', '--', '--', '--', ':', ':', ':', ':',]
plotter = render(print_states, plot_labels, line_styles, depth_plot_list=0, animate=0)



# DO ONE RANDOM EPISODE
plotter.clear()
state = env.reset()
first_state = np.concatenate((env.previous_state[0:6],env.ang,np.zeros(4)))
plotter.add(0,first_state)
done = False
t=0
while not done:
    t+=time_int_step
    action = policy.actor(torch.FloatTensor(state).to(device)).cpu().detach().numpy()
    state, _, done = env.step(action)
    plot_state = np.concatenate((env.state[0:6],env.ang,action))
    plotter.add(t,plot_state)
print('Env Solved, printing...')
plotter.plot()
# plotter.depth_plot()
an = animation()
an.animate(plotter.states)
plotter.clear()

