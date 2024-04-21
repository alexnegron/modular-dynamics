import torch 
import torch.nn as nn   
import numpy as np
relu = nn.ReLU()

torch.set_default_tensor_type(torch.DoubleTensor)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# torch.set_default_tensor_type(torch.FloatTensor)
# device = 'mps' if torch.backends.mps.is_built() else 'cpu'


class CTRNN(nn.Module):
    def __init__(self,
                 input_size,
                 device='cpu',
                 nNeurons=100,
                 dt=0.5,
                 tau=10.0,
                 ):

        super().__init__()

        self.device = device
        self.input_size = input_size
        self.nNeurons = nNeurons
        self.hidden_size = self.nNeurons
        self.dt = dt
        self.tau = tau
        self.alpha = dt / tau
        

        # Input to hidden layer
        self.input_to_hidden = nn.Linear(input_size, self.hidden_size, bias=False).to(device)
        nn.init.kaiming_uniform_(self.input_to_hidden.weight)

        # Hidden to hidden layer
        self.recurrent_weights = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size), requires_grad=True).to(device)
        nn.init.xavier_uniform_(self.recurrent_weights)
        
    def init_hidden(self):
        """
        Initialize the activities recurrent layer.
        """
        return torch.zeros(self.hidden_size, device=self.device)


    def recurrence(self, input, hidden):
        h2h = hidden @ self.recurrent_weights
        i2h = self.input_to_hidden(input)
        h_pre_act = i2h + h2h

        h_new = (1 - self.alpha) * hidden + self.alpha * relu(h_pre_act)
        return h_new

    def forward(self, input, hidden=None):
        if hidden is None: # initialize hidden neuron states
            hidden = self.init_hidden()

        # propagate input through ring module
        recurrent_acts = []
        steps = range(input.shape[0])
        for t in steps:
            hidden = self.recurrence(input[t, ...], hidden)

            # store ring network activity
            recurrent_acts.append(hidden)

        hidden_acts = torch.stack(recurrent_acts, dim=0)

        return hidden_acts,  hidden


class VanillaRNN(torch.nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()

        # continuous-time RNN (vanilla)
        self.CTRNN = CTRNN(input_size, **kwargs)

        # output from recurrent layer
        self.output = torch.nn.Linear(self.CTRNN.hidden_size, output_size, bias=True).to(device)
        # torch.nn.init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        activity, hidden_state = self.CTRNN(x)
        out = self.output(activity)

        return out, activity