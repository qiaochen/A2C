import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import ActorCriticNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = np.finfo(np.float32).eps.item()

class ACAgent:
    
    def __init__(self,
                 state_dim,                 # dimension of the state vector
                 action_dim,                # dimension of the action vector
                 num_envs,                  # number of parallel agents (20 in this experiment)
                 rollout_length=5,          # steps to sample before bootstraping
                 lr=1e-4,                   # learning rate
                 lr_decay=.95,              # learning rate decay rate
                 gamma=.99,                 # reward discount rate
                 value_loss_weight = 1.0,   # strength of value loss
                 gradient_clip = 5,         # threshold of gradient clip that prevent exploding
                 ):
        self.model = ActorCriticNetwork(state_dim, action_dim).to(device=device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lr = lr
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.total_steps = 0
        self.n_envs = num_envs
        self.value_loss_weight = value_loss_weight
        self.gradient_clip = gradient_clip
        self.rollout_length = rollout_length
        self.total_steps = 0
        
    def sample_action(self, state):
        """
        Sample action along with outputting the log probability and state values, given the states
        """
        state = torch.from_numpy(state).float().to(device=device)
        action, log_prob, state_value = self.model(state)
        return action, log_prob, state_value
    
    def update_model(self, experience):
        """
        Updates the actor critic network given the experience
        experience: list [[action,reward,log_prob,done,state_value]]
        """
        processed_experience = [None]* (len(experience) - 1)
        
        _advantage = torch.tensor(np.zeros((self.n_envs, 1))).float().to(device=device)   # initialize advantage Tensor
        _return = experience[-1][-1].detach()                                             # get returns
        for i in range(len(experience)-2,-1,-1):                                          # iterate from the last step
            _action, _reward, _log_prob, _not_done, _value = experience[i]                # get training data
            _not_done = torch.tensor(_not_done,device=device).unsqueeze(1).float()        # masks indicating the episodes not finished
            _reward = torch.tensor(_reward,device=device).unsqueeze(1)                    # get the rewards of the parallel agents
            _next_value = experience[i+1][-1]                                             # get the next states
            _return = _reward + self.gamma * _not_done * _return                          # compute discounted return
            _advantage = _reward + self.gamma * _not_done * _next_value.detach() - _value.detach() # advantage
            processed_experience[i] = [_log_prob, _value, _return,_advantage]
            
        log_prob, value, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_experience))
        policy_loss = -log_prob * advantages                                  # loss of the actor
        value_loss = 0.5 * (returns - value).pow(2)                           # loss of the critic (MSE)
        
        self.optimizer.zero_grad()
        (policy_loss + self.value_loss_weight * value_loss).mean().backward() # total loss
        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip) # clip gradient
        self.optimizer.step()
        
        self.total_steps += self.rollout_length * self.n_envs
        
        
    def act(self, state):
        """
        Conduct an action given input state vector
        Used in eveluation
        """
        state = torch.from_numpy(state).float().to(device=device)
        self.model.eval()
        action, _, _ = self.model(state)
        self.model.train()
        return action
    
    def save(self, path="./trained_model.checkpoint"):
        """
        Save state_dict of the model
        """
        torch.save({"state_dict":self.model.state_dict}, path)
        
    def load(self, path):
        """
        Load model and decay learning rate
        """
        state_dict = torch.load(path)['state_dict']
        self.model.load_state_dict(state_dict())
        
        # If recoverred for training, the learning rate is decreased
        self.lr *= self.lr_decay
        for group in self.optimizer.param_groups:
            group['lr'] = self.lr
            
            

        
        
    
        
        
