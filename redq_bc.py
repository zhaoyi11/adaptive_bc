import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EnsembleLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=7):
        super().__init__()

        self.ensemble_size = ensemble_size

        self.register_parameter('weight', nn.Parameter(torch.zeros(ensemble_size, in_features, out_features)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))

        nn.init.trunc_normal_(self.weight, std=1/(2*in_features**0.5))


    def forward(self, x):
        weight = self.weight
        bias = self.bias
        
        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_nets=10):
        super(Critic, self).__init__()

        self.nets = nn.Sequential(
            EnsembleLinear(state_dim + action_dim, 256, ensemble_size=num_nets),
            nn.ReLU(),
            EnsembleLinear(256, 256, ensemble_size=num_nets),
            nn.ReLU(),
            EnsembleLinear(256, 1, ensemble_size=num_nets)
        )


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        return self.nets(sa) # return dim: (num_nets, batch, 1)


class REDQ_BC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        num_nets=10,
        alpha=0.4,
        pretrain = True,
        use_q_min=False, # if False: REQD; True: min over Qs
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, num_nets=num_nets).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.num_nets = num_nets

        self.pretrain = pretrain
        self.use_q_min = use_q_min

        self.total_it = 0


    def select_action(self, state):
        # Return the action to interact with env.
        if len(state.shape) == 1: # if no batch dim
            state = state.reshape(1, -1)

        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)
        else:
            state = state.to(device)

        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().data.numpy().flatten()


    def train(self, data):
        self.total_it += 1

        state, action, next_state, reward, not_done = data

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            if self.use_q_min and not self.pretrain:
                target_Qs = self.critic_target(next_state, next_action) 
                target_Q,_ = torch.min(target_Qs, dim=0)
            else: # REDQ        
                random_idx = np.random.permutation(self.num_nets)
                target_Qs = self.critic_target(next_state, next_action)[random_idx] 
                target_Q1, target_Q2 = target_Qs[:2]

                target_Q = torch.min(target_Q1, target_Q2)

            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Qs = self.critic(state, action)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Qs.unsqueeze(0), target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic(state, pi).mean(0)

            actor_loss = -Q.mean() / Q.abs().mean().detach() + self.alpha * F.mse_loss(pi, action) 

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {"critic_loss": critic_loss.item(),
                "critic": current_Qs[0].mean().item()}

    def save(self, filename):
        torch.save({'critic': self.critic.state_dict(),
                    'critic_optimizer': self.critic_optimizer.state_dict(),
                    'actor': self.actor.state_dict(),
                    'actor_optimizer': self.actor_optimizer.state_dict(),
        }, filename + '_policy.pth')

    def load(self, filename):
        policy_dict = torch.load(filename + '_policy.pth')

        self.critic.load_state_dict(policy_dict['critic'])
        self.critic_optimizer.load_state_dict(policy_dict['critic_optimizer'])
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(policy_dict['actor'])
        self.actor_optimizer.load_state_dict(policy_dict['actor_optimizer'])
        self.actor_target = copy.deepcopy(self.actor)
