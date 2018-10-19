import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
torch.manual_seed(999)

def hidden_init(layer):
    """
    Used for parameter initialization
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorCriticNetwork(nn.Module):
    """
    The actor critic network
    The Actor and the Critic Share the same input encoder
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        
        # Actor head: output mean and std
        self.actor_fc = nn.Linear(128, 128)
        self.actor_out = nn.Linear(128, action_dim)
        
        self.std = nn.Parameter(torch.ones(1, action_dim))
        
        # critic head: output state value
        self.critic_fc = nn.Linear(128, 128)
        self.critic_out = nn.Linear(128, 1)
        self.reset_parameters()
        
    def forward(self, state):
        """
        Compute forward pass
        Input: state tensor
        Output: tuple of (clampped action, log probabilities, state values)
        """
        x = F.relu(self.fc1(state))
        mean = self.actor_out(F.relu(self.actor_fc(x)))
        dist = torch.distributions.Normal(mean, self.std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic_out(F.relu(self.critic_fc(x)))
        return torch.clamp(action, -1, 1), log_prob, value
    
    def reset_parameters(self):
        """
        Reset parameters to the initial states
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.actor_fc.weight.data.uniform_(*hidden_init(self.actor_fc))
        self.critic_fc.weight.data.uniform_(*hidden_init(self.critic_fc))
        self.actor_out.weight.data.uniform_(-3e-3, 3e-3)
        self.critic_out.weight.data.uniform_(-3e-3, 3e-3)

