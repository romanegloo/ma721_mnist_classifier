from torch import nn
import torch.nn.functional as F
from torch.nn.modules.module import _addindent
import numpy as np


# define a network with multiple layers
class DynamicNet(nn.Module):
    def __init__(self, args):
        super(DynamicNet, self).__init__()
        self.args = args
        self.input_layer = nn.Linear(args.input_dim, args.hidden_dim)
        self.middle_layer = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.output_layer = nn.Linear(args.hidden_dim, args.output_dim)

    def forward(self, x):
        h_relu = self.input_layer(x).clamp(min=0)
        for _ in range(self.args.num_hidden_layers - 1):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_layer(h_relu)
        return F.log_softmax(y_pred)


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