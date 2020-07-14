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
    PPO deep learning training algorithm. 
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        try:
            self.policy.load_state_dict(torch.load('./PPO_continuous_drone.pth',map_location=device))
            self.policy_old.load_state_dict(torch.load('./PPO_continuous_old_drone.pth',map_location=device))
            print('Saved models loaded')
        except:
            print('New models generated')
            pass
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())



def evaluate(env,agent,plotter,eval_steps=10):
    n_solved = 0
    rewards = 0
    time_steps = 0
    for i in range(eval_steps):
        state = env.reset()
        plotter.clear()
        done = False
        while True:
            time_steps += 1
            action = agent.policy.actor(torch.FloatTensor(state).to(device)).cpu().detach().numpy()
            state, reward, done = env.step(action)
            rewards += reward
            if i == eval_steps-1:
                plot_state = np.concatenate((env.state[0:5:2],env.ang,action))
                plotter.add(env.i*0.01,plot_state)
            if done:
                n_solved += env.solved
                break
    time_mean = time_steps/eval_steps
    solved_mean = n_solved/eval_steps        
    reward_mean = rewards/eval_steps
    plotter.plot()
    return reward_mean, time_mean, solved_mean
    
## HYPERPARAMETERS - CHANGE IF NECESSARY ##
lr = 0.0001
max_timesteps = 1000
action_std = 0.3
update_timestep = 4000
K_epochs = 80
T = 5


## HYPERPAREMETERS - PROBABLY NOT NECESSARY TO CHANGE ##
action_dim = 4
random_seed = 0
log_interval = 100
max_episodes = 100000
time_int_step = 0.01
solved_reward = 700
eps_clip = 0.2
gamma = 0.99
betas = (0.9, 0.999)
DEBUG = 0

# creating environment
env = quad(time_int_step, max_timesteps, euler=0, direct_control=1, deep_learning=1, T=T, debug=0)
state_dim = env.deep_learning_in_size
print_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
plot_labels = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'f1', 'f2', 'f3', 'f4']
line_styles = ['-', '-', '-', '--', '--', '--', ':', ':', ':', ':',]
plotter = render(print_states, plot_labels, line_styles, depth_plot_list=0, animate=0)

if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

memory = Memory()
ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
print(lr,betas)

# logging variables
t_since_last_plot = 0
running_reward = 0
avg_length = 0
time_step = 0
solved_avg = 0
eval_on_mean = True


# training loop
for i_episode in range(1, max_episodes+1):
    state = env.reset()
    for t in range(max_timesteps):
        t_since_last_plot += 1
        time_step +=1
        # Running policy_old:
        action = ppo.select_action(state, memory)
        state, reward, done = env.step(action)
    
        # Saving reward and is_terminals:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        
        # update if its time
        if time_step % update_timestep == 0:
            ppo.update(memory)
            memory.clear_memory()
            time_step = 0
        running_reward += reward
        if done:            
            break
    avg_length += t
    
    # save every 500 episodes
    if i_episode % 500 == 0:
        torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format('drone'))
        torch.save(ppo.policy_old.state_dict(), './PPO_continuous_old_{}.pth'.format('drone'))
        
    # logging
    if i_episode % log_interval == 0:
        reward_avg, time_avg, solved_avg = evaluate(env,ppo,plotter,20)
        avg_length = int(avg_length/log_interval)
        running_reward = int((running_reward/log_interval))
        print('Episode {} \t Avg length: {} \t Avg reward: {:.2f} \t Solved: {:.2f}'.format(i_episode, time_avg, reward_avg, solved_avg))
        running_reward = 0
        avg_length = 0
        
    # stop training if avg_reward > solved_reward
    if solved_avg > 0.95:
        reward_avg, time_avg, solved_avg = evaluate(env,ppo,plotter,200)
        if solved_avg > 0.95:
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format('drone'))
            torch.save(ppo.policy_old.state_dict(), './PPO_continuous_old_solved_{}.pth'.format('drone'))
            break
        