import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tasks.independent_integration_task import generate_dataset
import numpy as np

class IntegrationDataset(Dataset):
    def __init__(self,dim,batch_size,min_length,max_length,sample_length,
                 num_batches = 10,trajectory_type='binary',omega_value=0.05):
        self.dim = dim
        self.batch_size = batch_size
        self.min_length = min_length
        self.max_length = max_length
        self.length = np.random.randint(self.min_length,self.max_length) #current length
        self.sample_length = sample_length
        self.trajectory_type = trajectory_type
        self.test = False
        self.num_batches = num_batches
        self.omega_value = omega_value

        inputs,targets = generate_dataset(self.length,self.dim,self.trajectory_type,
                                          omega_value=self.omega_value)

        self.inputs = torch.tensor(inputs)
        self.targets = torch.tensor(targets)

    def set_test(self):
        self.test = True
        self.num_batches = 1
        self.sample_length = False

    def set_train(self):
        self.test = False
        self.sample_length = True

    def __len__(self):
        return self.batch_size*self.num_batches

    def __getitem__(self, idx):

        if self.sample_length:
            if idx % self.num_batches == 0:
                self.length = np.random.randint(self.min_length,self.max_length)
            
            length = self.length
                
        else:
            length = self.max_length

        # inputs,targets = generate_dataset(length,self.dim,self.trajectory_type,
        #                                   omega_value=self.omega_value)
        # return inputs, targets
        return self.inputs,self.targets