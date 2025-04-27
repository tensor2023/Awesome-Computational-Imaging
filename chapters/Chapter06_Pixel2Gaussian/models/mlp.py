import torch.nn as nn
import torch
import torch.nn.init as init

from models import register


@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, act='gelu'):
        super().__init__()
        if act is None:
            self.act = None
        elif act.lower() == 'relu':
            self.act = nn.ReLU()
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        else:
            assert False, f'activation {act} is not supported'
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            if self.act:
                layers.append(self.act)
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias) 
    
    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.contiguous().view(-1, x.shape[-1]))
        # x = torch.sigmoid(x)
        return x.view(*shape, -1)
