import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PreProcessing(nn.Module):
    # (batch, seq_len, state_dim)
    def __init__(self, state_dim, seq_len, dropout, d_model):
        super().__init__()

        self.linear = nn.Linear(state_dim, d_model)

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position=torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(2*torch.arange(0, d_model, 2).float()*(-math.log(10000))/d_model)

        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        
        pe = pe.unsqueeze(0) # to accommodate for multiple batches (1, seq_len, state_dim)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = self.linear(x)
        return self.dropout(x + self.pe[:, :x.shape[1], :].requires_grad_(False))
    

class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps:float=10**-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    # remember x has dimension (batch, seq_len, d_model)
    def forward(self, x):
        mu = torch.mean(x, dim=-1, keepdim=True) #batch, seq_len ,1
        std = torch.std(x, dim=-1, keepdim=True) #batch, seq_len ,1
        return self.alpha*((x - mu)/(std + self.eps)) + self.beta

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, seq_len, dropout, heads):
        super().__init__()
        self.d_model= d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.heads = heads

        assert d_model % heads == 0

        self.d_k =  d_model // heads

        self.W_k = nn.Linear(d_model, d_model, bias = False)
        self.W_q = nn.Linear(d_model, d_model, bias = False)
        self.W_v = nn.Linear(d_model, d_model, bias = False)
        self.W_o = nn.Linear(d_model, d_model, bias = False)
    
    # (batch, seq_len, state_dim)
    def forward(self, q, k, v, mask=None):
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)

        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2) #(batch, seq_len, h, d_k) -> (batch, h, seq_len, d_K)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)

        attention = (query @ key.transpose(-1, -2)) / math.sqrt(self.d_k) # (batch, h, seq_len, seq_len)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        attention = attention.softmax(dim=-1)

        x = attention @ value #(batch, h, seq_len, d_k)
        x = x.transpose(1,2).contiguous() #(batch, seq_len, h, d_k)
        x= x.view(x.shape[0], x.shape[1], self.heads*self.d_k) #(batch, seq_len, state_dim)

        return self.W_o(x)

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2  = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class ResidualConnection(nn.Module):
    def __init__(self, dropout, d_model):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, dropout, self_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock):
        super().__init__()
        self.dropout = dropout
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout, d_model) for _ in range(2)])
    def forward(self, x):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x))
        return self.residual_connections[1](x, self.feed_forward_block)


class Encoder(nn.Module):
    def __init__(self, layers, preprocessing: PreProcessing):
        super().__init__()
        self.layers = layers
        self.preprocessing = preprocessing
    def forward(self, x):
        x = self.preprocessing(x)
        for layer in self.layers:
            x = layer(x)
        return x

def BuildEncoder(state_dim, seq_len, dropout, d_model, d_ff, heads, N):
    preprocess = PreProcessing(state_dim, seq_len, dropout, d_model)
    self_attention_block = MultiHeadAttentionBlock(d_model, seq_len, dropout, heads)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

    layers = nn.ModuleList([])

    for _ in range(N):
        layers.append(EncoderBlock(d_model, dropout, self_attention_block, feed_forward_block))
    
    encoder  = Encoder(layers, preprocess)

    for p in encoder.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return encoder 

# For Now the X layer is just taking the last embedding of the sequence

class DQN(nn.Module):
    def __init__(self, state_dim, num_actions, num_hidden_1, num_hidden_2, d_model, d_ff, heads, N, dropout, seq_len):
        super().__init__()
        self.encoder = BuildEncoder(state_dim, seq_len, dropout, d_model, d_ff, heads, N)
      
        self.linear1 = nn.Linear(d_model, num_hidden_1)

        self.value_layer = nn.Linear(num_hidden_1, num_hidden_2)
        self.value = nn.Linear(num_hidden_2, 1)

        self.advantage_layer = nn.Linear(num_hidden_1, num_hidden_2)
        self.advantage = nn.Linear(num_hidden_2, num_actions)


    def forward(self, x):
        x = self.encoder(x)
        # Take the last state from the sequence (most recent state)
        x = x[:, -1, :]  # Shape: (batch, d_model)

        x = F.relu(self.linear1(x))

        v = F.relu(self.value_layer(x))
        a = F.relu(self.advantage_layer(x))

        a_out = self.advantage(a)

        q = self.value(v) + a_out - torch.mean(a_out)

        return q

        



    