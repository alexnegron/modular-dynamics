import torch.nn as nn
import torch
from torch.autograd import Variable
from multi_ring_modular_rnn import MultiModRNN

class RNNBase(nn.Module):
    def __init__(self,architecture,neurons,activation,input_dim,
                output_dim,seed,sigma,**rnn_kwargs):
        '''
        Initializes RNN from config data
        '''
        super(RNNBase,self).__init__()
        self.architecture = architecture
        self.input_dim = input_dim
        self.hidden_size = neurons
        self.sigma = sigma #won't use yet
        self.activation = activation
        self.nonlinearity = nn.ReLU() if activation == 'relu' else nn.Tanh()
        torch.manual_seed(seed) #set random seed

        self.rnn_kwargs = rnn_kwargs
        self.initialize_rnn(**rnn_kwargs)
        self.out = nn.Linear(self.hidden_size,output_dim)
        for w in self.parameters():
            if len(w.shape) == 1:
                nn.init.zeros_(w) #fill bias with 0s initially
            else:
                nn.init.xavier_uniform_(w, gain=1.0)

    def initialize_rnn(self,**rnn_kwargs):
        args = [self.input_dim,self.hidden_size]
        kwargs = rnn_kwargs
        if self.architecture == 'LSTM':
            self.rnn = LSTM #nn.LSTM
            #kwargs['batch_first'] = True
            kwargs['nonlinearity'] = self.nonlinearity

        elif self.architecture == 'GRU':
            self.rnn = GRU #nn.GRU
            #kwargs['batch_first'] = True
            kwargs['nonlinearity'] = self.nonlinearity

        elif self.architecture == 'VanillaRNN':
            self.rnn = VanillaRNN #nn.RNN
            #kwargs['batch_first'] = True
            kwargs['nonlinearity'] = self.nonlinearity

        elif self.architecture == 'UGRNN':
            self.rnn = UGRNN
            kwargs['nonlinearity'] = self.nonlinearity

        elif self.architecture == "ModularRNN":
            self.rnn = MultiModRNN

        self.rnn = self.rnn(*args, **kwargs)
        
    def forward(self,inputs,hidden=None):
        if hidden is None:
            hidden = Variable(inputs.new_zeros(1,inputs.size(0), self.hidden_size))
        if self.architecture == 'LSTM':
            if not isinstance(hidden,tuple): #if we don't get an initial cell state
                cell = Variable(inputs.new_zeros(1,inputs.size(0), self.hidden_size))
            hidden = (hidden,cell)
        hidden, _ = self.rnn(inputs,hidden)

        #hidden = self.nonlinearity(hidden)
        out = self.out(hidden)    
        return hidden, out   

class RNN(nn.Module):
    def __init__(self, input_dim,hidden_size,nonlinearity,n=2):
        super(RNN,self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.n = n
        self.x2h = nn.Linear(input_dim, n * hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, n * hidden_size, bias=True)

    def forward(self,inputs,hidden=None):
        hiddens = []
        for time in range(inputs.shape[1]):
            hidden = self.forward_pass(inputs[:,time],hidden)
            hiddens.append(hidden)
        hiddens = torch.cat(hiddens,dim=0)
        hiddens = torch.permute(hiddens, (1, 0, 2))
        return hiddens, None

class VanillaRNN(RNN):
    def __init__(self,input_dim,hidden_size,nonlinearity):
        super(VanillaRNN,self).__init__(input_dim,hidden_size,nonlinearity,n=1)

    def forward_pass(self,input,hidden):
        x_t = self.x2h(input)
        h_t = self.h2h(hidden)
        hidden = self.nonlinearity(x_t + h_t)
        return hidden

class UGRNN(RNN):
    def __init__(self,input_dim,hidden_size,nonlinearity):
        super(UGRNN,self).__init__(input_dim,hidden_size,nonlinearity,n=2)
        
    def forward_pass(self,input,hidden):
        x_t = self.x2h(input)
        h_t = self.h2h(hidden) 
        if len(h_t.shape) == 3:
            h_t = h_t.squeeze(1) #only remove one of the 1 dimensions, not both!
        x_c, x_g = x_t.chunk(2, 1)
        h_c, h_g = h_t.chunk(2, -1) #chunk by the hidden size dimensions
        new_h = self.nonlinearity(x_c + h_c)
        update_gate = torch.sigmoid(x_g + h_g)
        hidden = update_gate * hidden + (1 - update_gate) * new_h

        return hidden

##we need to write our own class for LSTMs and GRUs if we want to use ReLU
class GRU(RNN):
    def __init__(self,input_dim,hidden_size,nonlinearity):
        super(GRU,self).__init__(input_dim,hidden_size,nonlinearity,n=3)

    def forward_pass(self,input,hidden):
        x_t = self.x2h(input)
        h_t = self.h2h(hidden).squeeze()
        dimx = 1 if len(x_t.shape) > 1 else 0
        dimh = 1 if len(h_t.shape) > 1 else 0 
        x_reset, x_upd, x_new = x_t.chunk(3, dimx)
        h_reset, h_upd, h_new = h_t.chunk(3, dimh)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = self.nonlinearity(x_new + (reset_gate * h_new))

        hy = update_gate * hidden + (1 - update_gate) * new_gate

        return hy

class LSTM(RNN):
    def __init__(self,input_dim,hidden_size,nonlinearity):
        super(LSTM,self).__init__(input_dim,hidden_size,nonlinearity,n=4)

    def forward(self,inputs,hidden=None):
        hiddens = []
        for time in range(inputs.shape[1]):
            hidden = self.forward_pass(inputs[:,time],hidden)
            hiddens.append(hidden[0]) #only append hidden state
        hiddens = torch.cat(hiddens,dim=0)
        hiddens = torch.permute(hiddens, (1, 0, 2))
        return hiddens, None

    def forward_pass(self,input,hidden):
        hidden, cell = hidden

        x_t = self.x2h(input)
        h_t = self.h2h(hidden).squeeze()
        gates = x_t + h_t
        # Get gates (i_t, f_t, g_t, o_t)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.sigmoid(cell_gate)
        o_t = torch.sigmoid(output_gate)

        cy = cell * f_t + i_t * g_t

        hy = o_t * self.nonlinearity(cy)

        return hy, cy