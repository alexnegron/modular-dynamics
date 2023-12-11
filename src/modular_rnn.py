import torch 
import torch.nn as nn   
import numpy as np
from src.utils import generate_alpha_matrix
relu = nn.ReLU()

torch.set_default_tensor_type(torch.DoubleTensor)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RingModule(nn.Module):
    def __init__(self,
                 input_size,
                 device='cpu',
                 nNeurons=100,
                 nBumps=1,
                 length=40,
                 gNoiseMag=None,
                 fano=None,
                 dt=0.5,
                 tau=10.0,
                 alpha = 1.0,
                 wWeight=8.0,
                 wShift=2,
                 wScaling=True,
                 restingMag=1.0,
                 gammaMultiplier=1.0,
                 pulseMag = 1.0,
                 pulsePosition = 50
                 ):

        super().__init__()

        self.device = device
        self.input_size = input_size
        self.nNeurons = nNeurons
        self.hidden_size = 2 * self.nNeurons
        self.nBumps = nBumps
        self.dt = dt
        self.length = length
        self.tau = tau
        self.alpha = dt / tau
        self.phiFunction = nn.ReLU()
        self.wAttractor = generate_alpha_matrix(nNeurons,
                                                length,
                                                alpha,
                                                wWeight,
                                                wShift,
                                                wScaling,
                                                device=device).to(device)

        self.gNoiseMag = gNoiseMag
        self.fano = fano
        self.restingMag = restingMag
        self.gammaMultiplier = gammaMultiplier

        self.pulseMag = pulseMag
        self.pulsePosition = pulsePosition

        # Setup inputs to ring
        self.input_to_vel = nn.Linear(input_size, 1, bias=False).to(device) # learnable
        nn.init.ones_(self.input_to_vel.weight) 
        # important! note that this initialization, if random, will cause the hidden activity 
        # initialization to also be random, since the init_hidden() involves running the recurrent dynamics
        # which calls self.input_to_vel 

        self.vel_to_ring = nn.Linear(1, self.hidden_size, bias=True).to(device) # unlearnable
        self.vel_to_ring.weight.requires_grad = False
        self.vel_to_ring.bias.requires_grad = False # fix the bias during training
        self.gamma = (  # Coupling strength between 'velocity' neuron and network
            gammaMultiplier * torch.cat((-torch.ones(self.nNeurons), torch.ones(self.nNeurons))).to(device)
        ).unsqueeze(1)
        self.vel_to_ring.weight.data.copy_(self.gamma); # set coupling from 'velocity' neuron to ring network manually
        self.vel_to_ring.bias.copy_(torch.ones((self.hidden_size,))); # bias is fixed at 1

    def init_hidden(self):
        """
        Initialize the activities in each ring.

        - pulsePosition is chosen to inject a delta pulse of activity at the specified neuron index in each ring.
        - Dynamics are run until bumps stabilize at these positions on each ring.
        - The resulting activities form rings' activity at initialization.
        """

        bump_period = int(self.nNeurons / self.nBumps) # bump distance
        pulse_inds = bump_period * np.arange(self.nBumps)
        pulse_inds = np.concatenate((pulse_inds, self.nNeurons + pulse_inds))

        # pulsePosition = 12 # !important! for now, bumps will always be initialized at the 12th neuron in each ring

        pulse_inds += int(self.pulsePosition % bump_period)
        pulse_inputs = torch.zeros(2 * self.nNeurons, device=self.device)
        pulse_inputs[pulse_inds] = self.pulseMag

        # hidden = 0.005 * torch.rand(2 * self.nNeurons, device=self.device)
        # hidden = 0.05 * torch.rand(2 * self.nNeurons, device=self.device)
        hidden = 0.005 * torch.ones(2 * self.nNeurons, device=self.device) + pulse_inputs

        tSetup = 1_000

        # init_drive = torch.tensor([.1,1,0]).double().to(self.device) # some arbitrary drive to initialize
        init_drive = torch.tensor([.1]).double().to(self.device) # some arbitrary drive to initialize

        for t in np.arange(0, tSetup): # run dynamics a little
            hidden = self.recurrence(init_drive, hidden)

        while (torch.argmax(hidden[:self.nNeurons]) - self.pulsePosition != 0):
            hidden = self.recurrence(init_drive, hidden)

        # print('bumps initialized')

        return hidden


    def recurrence(self, input, hidden):
        # h2h = torch.matmul(self.wAttractor, hidden.T)
        h2h = hidden @ self.wAttractor
        i2h = self.vel_to_ring(self.input_to_vel(input))
        h_pre_act = i2h + h2h

        h_new = (1 - self.alpha)*hidden + self.alpha*relu(h_pre_act)
        return h_new

    def forward(self, input, hidden=None):
        if hidden is None: # initialize ring neuron states
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


class ModRNN(torch.nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()

        # continuous-time RNN with fixed ring module
        self.CTRNN = RingModule(input_size, **kwargs)

        # output from recurrent layer
        self.output = torch.nn.Linear(self.CTRNN.hidden_size, output_size, bias=True).to(device)
        # torch.nn.init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        activity, hidden_state = self.CTRNN(x)
        out = self.output(activity)

        return out, activity