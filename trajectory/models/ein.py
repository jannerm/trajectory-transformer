import math
import torch
import torch.nn as nn
import pdb

class EinLinear(nn.Module):

    def __init__(self, n_models, in_features, out_features, bias):
        super().__init__()
        self.n_models = n_models
        self.out_features = out_features
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(n_models, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_models, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.n_models):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, input):
        """
            input : [ B x n_models x input_dim ]
        """
        ## [ B x n_models x output_dim ]
        output = torch.einsum('eoi,bei->beo', self.weight, input)
        if self.bias is not None:
            raise RuntimeError()
        return output

    def extra_repr(self):
        return 'n_models={}, in_features={}, out_features={}, bias={}'.format(
            self.n_models, self.in_features, self.out_features, self.bias is not None
        )
