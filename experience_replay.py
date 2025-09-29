from collections import deque
from re import S
import torch
import torch.nn as nn
import random


class Replay:
    def __init__(self, maxlen):
        self.experience = []
        self.maxlen= maxlen
        self.size = 0
        self.weights = []

    def push(self, transitions):
        self.experience.append(transitions)
        self.size += len(transitions)
        self.weights.append(len(transitions))
        if self.size > self.maxlen:
            self.size -= len(self.experience[0])
            self.experience.pop(0)
            self.weights.pop(0)
        
    def sample(self, sample_size, state_dim, seq_len, device):

        row_idx = random.choices(range(0, len(self.weights)), self.weights, k=sample_size)
        actions = []
        rewards = []
        dones= []
        states = []
        next_states = []
        for i in range(len(row_idx)):
            col_idx = random.randrange(len(self.experience[row_idx[i]]))
            sample = self.experience[row_idx[i]][max(0,col_idx-seq_len+1):col_idx+1]
            num_pad = max(0,seq_len-len(sample))

            actions.append(sample[-1][1])
            rewards.append(sample[-1][3])
            dones.append(sample[-1][4])


            s = []
            #add padding to states that start at a position less than 16
            for _ in range(num_pad):
                s.append(torch.zeros(state_dim, device=device))
            for j in range(len(sample)):
                s.append(sample[j][0])
            
            s = torch.stack(s, dim=0)
            assert len(s) == seq_len
            states.append(s)
            
            next_s = []
            for _ in range(num_pad):
                next_s.append(torch.zeros(state_dim, device=device))
            if len(sample) < seq_len:
                next_s[-1] = sample[0][0]
            for j in range(len(sample)):
                next_s.append(sample[j][2])

            next_s = torch.stack(next_s, dim=0)
            assert len(next_s) == seq_len
            next_states.append(next_s)

        return torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(next_states), torch.tensor(dones).float().to(device)

    def __len__(self):
        return self.size
