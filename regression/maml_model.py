"""
Neural network models for the regression experiments
"""
import math

import torch
import torch.nn.functional as F
from torch import nn



class TransformerNet(nn.Module):
     def __init__(self, n_inputs, n_hidden, size_hidden, n_o, device):
        super(TransformerNet,self).__init__()
        #self.task_context = torch.zeros(size_hidden).to(device)
       # self.task_context.requires_grad = True #maybe break net into two, similar to CAVIA
        self.net=nn.Sequential(OrderedDict([
            ('l1',nn.Linear(n_inputs, size_hidden)),
            ('relu1',nn.ReLU()),
            ('l2',nn.Linear(size_hidden, size_hidden)),
            ('relu2',nn.ReLU()),
            ('l3',nn.Linear(size_hidden, size_hidden)),
        ]))
    

     def forward(self,x):
     #   if len(self.task_context) != 0:
      #      x = torch.cat((x, self.task_context.expand(x.shape[0], -1)), dim=1)
       # else:
        #    x = torch.cat((x, self.task_context))
        return self.net(x)

class FCNet(nn.Module):
     def __init__(self, n_inputs, n_hidden, size_hidden, n_out):
        super(FCNet, self).__init__()

        # fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(n_inputs, size_hidden))
        for k in range(n_hidden - 2):
            self.fc_layers.append(nn.Linear(size_hidden, size_hidden))
        self.fc_layers.append(nn.Linear(size_hidden, n_out))

        # context parameters (note that these are *not* registered parameters of the model!)
        #self.num_context_params = num_context_params
        #self.context_params = None
        #self.reset_context_params()

     def forward(self, x):

        # concatenate input with context parameters
        #x = torch.cat((x, self.context_params.expand(x.shape[0], -1)), dim=1)

        for k in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[k](x))
        y = self.fc_layers[-1](x)

        return y

def transformer_loss(L0, L1, model, device):
    params = [w for w in model.weights] + [b for b in model.biases] + [model.task_context]
    gradients0 = torch.autograd.grad(L0, params, retain_graph=True, create_graph=True)
    gradients1 = torch.autograd.grad(L1, params, retain_graph=True, create_graph=True)
    loss = torch.zeros(1).to(device)
    for i in range(len(gradients0)):
        a = torch.flatten(gradients0[i].clamp_(-10, 10)).to(device)
        b = torch.flatten(gradients1[i].clamp_(-10, 10)).to(device)
        product = torch.dot(a, b)
        loss = loss - torch.sum(product)
    divisor = norm_tensorlist(gradients0, device)*norm_tensorlist(gradients1, device)
    loss = torch.div(loss , divisor)
    return loss

def cosine_loss(L0, L1, model, device):
    params = [w for w in model.weights] + [b for b in model.biases] + [model.task_context]
    gradients0 = torch.autograd.grad(L0, params, retain_graph=True, create_graph=True)
    gradients1 = torch.autograd.grad(L1, params, retain_graph=True, create_graph=True)
    criterion = nn.CosineEmbeddingLoss().to(device)
    mask = torch.ones(1).to(device)
    views1 =[]
    views2 = []
    for i in range(len(gradients0)):
        view1 = gradients0[i].clamp_(-10, 10).view(-1, 1).to(device)
        view2 = gradients1[i].clamp_(-10, 10).view(-1, 1).to(device)
        views1.append(view1)
        views2.append(view2)
    grad0 = torch.cat(views1, 0).to(device)
    grad1 = torch.cat(views2, 0).to(device)
    loss = criterion(grad0, grad1, mask)
    return loss

def phi_gradients(model, device):
    listOfGradients = []
    for p in model.parameters():
        listOfGradients.append(torch.zeros(list(p.size())).to(device))
    return listOfGradients

def norm_tensorlist(T1, device):
    sum_powers_T1 = torch.zeros(1).to(device)
    for i in range(len(T1)):
        sum_powers_T1 = sum_powers_T1 + torch.sum(torch.pow(T1[i], 2))
    return torch.sqrt(sum_powers_T1)



#######----------------------------------------------------------------------------------------------------
##
##           Luisa's stuff
##
######-----------------------------------------------------------------------------------------------------
class MamlModel(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_weights,
                 num_context_params,
                 device
                 ):
        """
        :param n_inputs:            the number of inputs to the network
        :param n_outputs:           the number of outputs of the network
        :param n_weights:           for each hidden layer the number of weights
        :param num_context_params:  number of additional inputs (trained together with rest)
        """
        super(MamlModel, self).__init__()

        # initialise lists for biases and fully connected layers
        self.weights = []
        self.biases = []

        # add one
        self.nodes_per_layer = n_weights + [n_outputs]

        # additional biases
        self.task_context = torch.zeros(num_context_params).to(device)
        self.task_context.requires_grad = True

        # set up the shared parts of the layers
        prev_n_weight = n_inputs + num_context_params
        for i in range(len(self.nodes_per_layer)):
            w = torch.Tensor(size=(prev_n_weight, self.nodes_per_layer[i])).to(device)
            w.requires_grad = True
            self.weights.append(w)
            b = torch.Tensor(size=[self.nodes_per_layer[i]]).to(device)
            b.requires_grad = True
            self.biases.append(b)
            prev_n_weight = self.nodes_per_layer[i]

        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(len(self.nodes_per_layer)):
            stdv = 1. / math.sqrt(self.nodes_per_layer[i])
            self.weights[i].data.uniform_(-stdv, stdv)
            self.biases[i].data.uniform_(-stdv, stdv)

    def forward(self, x):

        if len(self.task_context) != 0:
            x = torch.cat((x, self.task_context.expand(x.shape[0], -1)), dim=1)
        else:
            x = torch.cat((x, self.task_context))

        for i in range(len(self.weights) - 1):
            x = F.relu(F.linear(x, self.weights[i].t(), self.biases[i]))
        y = F.linear(x, self.weights[-1].t(), self.biases[-1])

        return y
