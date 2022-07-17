import numpy as np
import torch.nn as nn
import torch
class MLP(nn.Module):
    
    def __init__(self, in_dim=3, out_dim=2, hidden_dims=[], use_bias=True):

        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # If we have no hidden layer, just initialize a linear model (e.g. in logistic regression)
        if len(hidden_dims) == 0:
            layers = [nn.Linear(in_dim, out_dim, bias=use_bias)]
        else:
        # 'Actual' MLP with dimensions in_dim - num_hidden_layers*[hidden_dim] - out_dim
            layers = [nn.Linear(in_dim, hidden_dims[0], bias=use_bias), nn.ReLU()]

        # Loop until before the last layer
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            layers += [nn.Linear(hidden_dim, hidden_dims[i + 1], bias=use_bias),
                    nn.ReLU()]

        # Add final layer to the number of classes
        layers += [nn.Linear(hidden_dims[-1], out_dim, bias=use_bias)]
        
        self.main = nn.Sequential(*layers)

    def soft_argmax(vec):
        pass

    def forward(self, X):
        # dimension of hidden_output is steps*2
        # dimension of output is steps*1
        hidden_output = self.main(X)
        # softmax=nn.Softmax(dim=1)
        # hidden_output=softmax(hidden_output)
        output=torch.ones(X.size()[0])
        # need to convert hidden_output to output by soft-argmax
        return output