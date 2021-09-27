# Some of the code was taken from - https://github.com/WendyShang/flare/blob/main/encoder.py
# Full code has been checked.

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math, random
import cv2
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import matplotlib.pyplot as plt


# <h3>Use Cuda</h3>

# In[3]:
seed = 7779


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
# <h2>Prioritized Replay Buffer</h2>

# <p>Prioritized Experience Replay: https://arxiv.org/abs/1511.05952</p>

# In[28]:

from collections import deque
# <h2>Prioritized Replay Buffer</h2>

# <p>Prioritized Experience Replay: https://arxiv.org/abs/1511.05952</p>

# In[28]:


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        batch       = list(zip(*samples))
        states      = np.concatenate(batch[0])
        actions     = batch[1]
        rewards     = batch[2]
        next_states = np.concatenate(batch[3])
        dones       = batch[4]
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

# <h2>Computing Temporal Difference Loss</h2>

# In[14]:


def compute_td_loss(batch_size, beta):
    state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size, beta) 

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))
    weights    = Variable(torch.FloatTensor(weights))

    q_values      = current_model(state)
    next_q_values = target_model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss  = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss  = loss.mean()
        
    optimizer.zero_grad()
    loss.backward()
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()
    
    return loss


# <h1>Atari Environment</h1>

# In[17]:


from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch


# In[18]:


env_id = "FreewayNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)
env.seed(seed)
env.action_space.seed(seed)

# In[36]:
T = env.observation_space.shape[0]
print("The number of time steps = ", T)
#Done - Checked my code with the paper's using code difference checker
class CnnDQN(nn.Module):
    def __init__(self, obs_shape, feature_dim,
                 num_filters=32,
                 output_logits=False,
                 num_layers=2,
                 image_channel=1):
        super(CnnDQN, self).__init__()

        assert len(obs_shape) == 3

        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.image_channel = image_channel
        #It is assumed that the input has stacked images. For instance, if the input has 3 channels and 4 stacked images, the first dimension should be 12.
        #Since we only have a grayscale channel, time step should be 3.
        time_step = obs_shape[0] // self.image_channel

        self.convs = nn.ModuleList(
            [nn.Conv2d(self.image_channel, num_filters, 3, stride=2)]
        )
        self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        for i in range(2, num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.outputs = dict()
        # Shape of x =  torch.Size([32, 3, 84, 84])
        x = torch.randn([32] + list(obs_shape))

        self.out_dim = self.forward_conv(x, flatten=False).shape[-1]


        self.fc = nn.Linear(num_filters * self.out_dim * self.out_dim * (2 * time_step - 2), self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.output_logits = output_logits
        # self.features = nn.Sequential(
        #     nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )
        #
        # self.fc = nn.Sequential(
        #     nn.Linear(self.feature_size(), 512),
        #     nn.ReLU(),
        #     nn.Linear(512, self.num_actions)
        # )

    def forward_conv(self, obs, flatten=True):
        # Checking obs.max() and normalizing it was drastically slowing down the algorithm. Also I do not normalize in other environments.
        # if obs.max() > 1.:
        #     obs = obs / 255.



        # Since we only have a grayscale channel, time step should be 3. The first channel of obs will be the batch size.
        # Therefore the second channel of obs will have to be used to get time steps.
        time_step = obs.shape[1] // self.image_channel
        obs = obs.view(obs.shape[0], time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
        #todo Why do I have to change view to reshape
        obs = obs.reshape(obs.shape[0] * time_step, self.image_channel, obs.shape[-2], obs.shape[-1])

        #I honestly don't see the purpose of some of these dictionary values. Maybe used by the original author for debugging.

        self.outputs['obs'] = obs
        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        conv = torch.relu(self.convs[1](conv))
        self.outputs['conv%s' % (1 + 1)] = conv

        for i in range(2, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        #Size is another name for shape in Torch
        # Bringing conv to the shape of (batch size, time steps, channels, height/width, height/width)
        conv = conv.view(conv.size(0) // time_step, time_step, conv.size(1), conv.size(2), conv.size(3))

        # This should extract out all the time steps starting from the second one.
        conv_current = conv[:, 1:, :, :, :]
        # The second term in the RHS contains time steps starting from the first except the last.
        # Therefore conv_prev contains the latent difference between each succeeding time frame. Also, the detach over here is important.
        conv_prev = conv_current - conv[:, :time_step - 1, :, :, :].detach()
        # Concatentating the encoded images with its latent difference
        conv = torch.cat([conv_current, conv_prev], axis=1)
        conv = conv.view(conv.size(0), conv.size(1) * conv.size(2), conv.size(3), conv.size(4))

        if not flatten:
            return conv
        else:
            conv = conv.view(conv.size(0), -1)

            return conv


    def forward(self, obs, detach=False):

        h = self.forward_conv(obs)
        if detach:
            h = h.detach()
        try:
            h_fc = self.fc(h)
        except:
            print(obs.shape)
            print(h.shape)
            assert False
        self.outputs['fc'] = h_fc
        #Doing a layer norm over here. The layer norm cannot be removed. Read the paper for details.
        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        # This should be false
        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out
    

    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action


# In[37]:


current_model = CnnDQN(env.observation_space.shape, env.action_space.n)
target_model  = CnnDQN(env.observation_space.shape, env.action_space.n)

if USE_CUDA:
    current_model = current_model.cuda()
    target_model  = target_model.cuda()
    
optimizer = optim.Adam(current_model.parameters(), lr=0.0001)

replay_initial = 10000
replay_buffer  = NaivePrioritizedBuffer(100000)

update_target(current_model, target_model)


# <h3>Epsilon greedy exploration</h3>

# In[38]:


epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


# <h3>Beta Prioritized Experience Replay</h3>

# In[40]:


beta_start = 0.4
beta_frames = 100000
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)


# <h3>Training</h3>

# In[42]:


num_frames = 1400000
batch_size = 32
gamma      = 0.99

losses = []
reward_step = np.empty(shape = num_frames)
all_rewards = []
episode_reward = 0

state = env.reset()
print("Shape of state = ", state.shape)
filename = "flare_"+env_id[0:6]+"_"+str(seed)+"_"+str(T)+".out"

for frame_idx in range(1, num_frames + 1):
    print("Frame = ", frame_idx)
    epsilon = epsilon_by_frame(frame_idx)
    action = current_model.act(state, epsilon)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    reward_step[frame_idx - 1] = reward

    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        np.savetxt(filename, all_rewards, delimiter=',')
        episode_reward = 0
        
    if len(replay_buffer) > replay_initial:
        beta = beta_by_frame(frame_idx)
        loss = compute_td_loss(batch_size, beta)
        losses.append(loss.data)

    if frame_idx % 100000 == 0:
        print("Frame Index = ", frame_idx)
        np.savetxt('pr_tsm_step.out', reward_step, delimiter=',')

    if frame_idx % 1000 == 0:
        update_target(current_model, target_model)

    if frame_idx % 1400000 == 0:
        print("Simulation " + filename + " Complete !")
        

