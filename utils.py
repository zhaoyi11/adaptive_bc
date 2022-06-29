import numpy as np
import torch
import gym
import d4rl

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, states, actions, next_states, rewards, dones):
        for state, action, next_state, reward, done in zip(states, actions, next_states, rewards, dones):
            self.add(state, action, next_state, reward, done)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def convert_D4RL(self, dataset):
        data_size = int(dataset['observations'].shape[0]) # only use maximal 100_000 data points 
        if data_size < self.max_size:
            self.state[:data_size] = dataset['observations'][-data_size:]
            self.action[:data_size] = dataset['actions'][-data_size:]
            self.next_state[:data_size] = dataset['next_observations'][-data_size:]
            self.reward[:data_size] = dataset['rewards'][-data_size:].reshape(-1,1)
            self.not_done[:data_size] = 1. - dataset['terminals'][-data_size:].reshape(-1,1)
        else:
            self.max_size = data_size
            self.state = dataset['observations'][-data_size:]
            self.action = dataset['actions'][-data_size:]
            self.next_state = dataset['next_observations'][-data_size:]
            self.reward = dataset['rewards'][-data_size:].reshape(-1,1)
            self.not_done = 1. - dataset['terminals'][-data_size:].reshape(-1,1)
            
        self.size = data_size
        self.ptr = data_size % self.max_size	

    def distill(self, dataset, env_name, sample_method, ratio=0.05):
        # random distill dataset, keep at least 50_000 data points.
        data_size = max(int(ratio * dataset['observations'].shape[0]), 50_000) # at least keep 50_000 data

        if sample_method == "random":
            ind = np.random.randint(0, dataset['observations'].shape[0], size=data_size) 
        elif sample_method == "best": # select the last data
            if env_name == 'hopper-medium-expert-v0':  # this dataset is expert + replay
                ind = np.arange(0, data_size)
            else:
                full_data_size = dataset['observations'].shape[0]
                ind = np.arange(full_data_size - data_size, full_data_size)
        elif sample_method == "policy":
            ind = self._ind_fit_policy()
        else:
            raise ValueError

        if data_size < self.max_size:
            self.state[:data_size] = dataset['observations'][ind]
            self.action[:data_size] = dataset['actions'][ind]
            self.next_state[:data_size] = dataset['next_observations'][ind]
            self.reward[:data_size] = dataset['rewards'][ind].reshape(-1,1)
            self.not_done[:data_size] = 1. - dataset['terminals'][ind].reshape(-1,1) 
        else:
            self.max_size = data_size
            self.state = dataset['observations'][ind]
            self.action = dataset['actions'][ind]
            self.next_state = dataset['next_observations'][ind]
            self.reward = dataset['rewards'][ind].reshape(-1,1)
            self.not_done = 1. - dataset['terminals'][ind].reshape(-1,1) 

        self.size = data_size
        self.ptr = data_size % self.max_size

    def get_dataset_stats(self, dataset):
        episode_returns = []
        returns = 0
        epi_length = 0
        for r, d in zip(dataset['rewards'], dataset['terminals']):
            if d:
                episode_returns.append(returns)
                returns = 0
                epi_length = 0
            else:
                epi_length += 1
                returns += r
                if epi_length == 1000:
                    episode_returns.append(returns)
                    returns = 0
                    epi_length = 0
        episode_returns = np.array(episode_returns)
        return episode_returns.mean(), episode_returns.std()

    def normalize_states(self, eps = 1e-3):
        mean = self.state.mean(0,keepdims=True)
        std = self.state.std(0,keepdims=True) + eps
        self.state = (self.state - mean)/std
        self.next_state = (self.next_state - mean)/std
        return mean, std

    def get_all(self,):
        return (
            torch.FloatTensor(self.state[:self.size]).to(self.device),
            torch.FloatTensor(self.action[:self.size]).to(self.device),
            torch.FloatTensor(self.next_state[:self.size]).to(self.device),
            torch.FloatTensor(self.reward[:self.size]).to(self.device),
            torch.FloatTensor(self.not_done[:self.size]).to(self.device)
        )

    def random_split(self, val_size):
        """ Return training batch and validation batch. Training and validation data are splited randomly."""
        data_size = self.size
        permutation = np.random.permutation(data_size)
        
        training_batch = (torch.FloatTensor(self.state[permutation[val_size:]]).to(self.device),
                            torch.FloatTensor(self.action[permutation[val_size:]]).to(self.device),
                            torch.FloatTensor(self.next_state[permutation[val_size:]]).to(self.device),
                            torch.FloatTensor(self.reward[permutation[val_size:]]).to(self.device),
                            torch.FloatTensor(self.not_done[permutation[val_size:]]).to(self.device)
                        )		
            
        validation_batch = (torch.FloatTensor(self.state[permutation[:val_size]]).to(self.device),
                            torch.FloatTensor(self.action[permutation[:val_size]]).to(self.device),
                            torch.FloatTensor(self.next_state[permutation[:val_size]]).to(self.device),
                            torch.FloatTensor(self.reward[permutation[:val_size]]).to(self.device),
                            torch.FloatTensor(self.not_done[permutation[:val_size]]).to(self.device)
                        )						

        return training_batch, validation_batch

    def _ind_good_trajectories(self,):
        pass

    def _ind_fit_policy(self,):
        pass


########## Helper functions ##########
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    avg_length = 0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            avg_length += 1

    avg_reward /= eval_episodes
    avg_length = int(avg_length / eval_episodes)

    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    return {'d4rl': d4rl_score, 'evaluation': avg_reward, 'length': avg_length}

def rollout(rollout_batch_size, rollout_horizon, transition, policy, env_buffer, model_buffer, expl_noise=0.1, max_action=1):
    """ Rollout the learned dynamic model to generate imagined transitions. """
    states = env_buffer.sample(rollout_batch_size)[0]  # obs is tensor.cuda
    steps_added = []
    for h in range(rollout_horizon):
        with torch.no_grad():
            action = policy.actor(states, deterministic=False, with_logprob=False)
            # input is tensor.device, we hope the outputs are also tensor.device
            next_states, r, done, info = transition.step(states, action, use_penalty=False) # shape of r and done: [batch], no additional dim.
        model_buffer.add_batch(states.cpu().numpy(), action.cpu().numpy(),next_states.cpu().numpy(), r.cpu().numpy(), done.cpu().numpy())

        steps_added.append(states.shape[0])          
        nonterm_mask = torch.logical_not(done)
        if nonterm_mask.sum() == 0:
            print('[ Model Rollout ] Breaking early: {} | {} / {}'.format(h, nonterm_mask.sum(), nonterm_mask.shape))
            break

        states = next_states[nonterm_mask]

    mean_rollout_length = sum(steps_added) / rollout_batch_size
    return {"Rollout": mean_rollout_length}

def process_sac_data(env_buffer, model_buffer, batch_size, real_ratio):
    """ Cat samples from env_buffer and model_buffer. And suqeeze the r and d (sac update assume no additional dim)
    The returned data format: (o, a, r, d, n_o), and each entry with shape: [env_batch+model_batch, ...]
    """
    env_batch_size = int(batch_size * real_ratio)
    model_batch_size = batch_size - env_batch_size

    env_batch = env_buffer.sample(env_batch_size) # sampled data format: (o, a, r, d, n_o), o.shape: [batch, ...], r.shape: [batch]
    model_batch = model_buffer.sample(model_batch_size)

    batch = [torch.cat((env_item, model_item), dim=0) for env_item, model_item in zip(env_batch, model_batch)] # cat among batch_size
    
    return batch

def compute_num_grad(timesteps, min_gradsteps=10, max_gradsteps=40, min_timesteps=0, max_timesteps=10_000):
    return int(min(max(min_gradsteps + (timesteps - min_timesteps)/(max_timesteps - min_timesteps)*(max_gradsteps - min_gradsteps),
                        min_gradsteps),
                    max_gradsteps))

def compute_min_q_weight(timesteps, min_w, max_w, min_t, max_t):
    return max_w - float(min(max(min_w + (timesteps - min_w)/(max_t - min_t)*(max_w - min_w),
                        min_w),
                    max_w))