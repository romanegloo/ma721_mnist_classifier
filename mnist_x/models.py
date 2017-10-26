from torch import nn
import torch.nn.functional as F
from torch.nn.modules.module import _addindent
import torch.nn.init as init
import numpy as np
from functools import partial


# define a network with multiple layers
class DynamicNet(nn.Module):
    def __init__(self, args):
        super(DynamicNet, self).__init__()
        self.args = args
        self.input_bn = nn.BatchNorm1d(args.input_dim)
        self.input_layer = nn.Linear(args.input_dim, args.num_hidden_units)
        self.middle_bn = nn.BatchNorm1d(args.num_hidden_units)
        self.middle_layers = nn.ModuleList()
        for _ in range(args.num_hidden_layers - 1):
            self.middle_layers.append(nn.Linear(args.num_hidden_units,
                                                args.num_hidden_units))
        self.output_layer = nn.Linear(args.num_hidden_units, args.output_dim)

        # initialize weights
        self.initialize_weights(args.weight_init)

    def forward(self, x):
        h_relu = self.input_layer(self.input_bn(x)).clamp(min=0)
        for i in range(self.args.num_hidden_layers - 1):
            h_relu = self.middle_layers[i](self.middle_bn(h_relu)).clamp(min=0)
        y_pred = self.output_layer(h_relu)
        return F.log_softmax(y_pred)

    def initialize_weights(self, fn_name):
        if fn_name == 'uniform':
            fn = init.uniform
        elif fn_name == 'normal':
            fn = init.normal
        elif fn_name == 'xavier_normal':
            fn = partial(init.xavier_normal, gain=np.sqrt(2))
        else:
            return

        fn(self.input_layer.weight)
        for layer in self.middle_layers:
            fn(layer.weight)
        fn(self.output_layer.weight)


# ------------------------------------------------------------------------------
# Utils
# - torch_summarize: displays the summary note with weights and parameters of
#  the network (obtained from http://bit.ly/2glYWVV)
# ------------------------------------------------------------------------------


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    total_params = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            nn.modules.container.Container,
            nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        total_params += params
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr, total_params