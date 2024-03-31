import torch
import torch.nn as nn
from torch.utils.data import Dataset
from independent_integration_task import generate_dataset
import numpy as np

class IntegrationDataset(Dataset):
    def __init__(self,dim,batch_size,min_length,max_length,sample_length,trajectory_type='binary'):
        self.dim = dim
        self.batch_size = batch_size
        self.min_length = min_length
        self.max_length = max_length
        self.sample_length = sample_length
        self.trajectory_type = self.trajectory_type

    def test(self):
        self.sample_length = False

    def train(self):
        self.sample_length = True

    def __len__(self):
        #put a fixed number here just for completeness
        return self.batch_size*100

    def __getitem__(self, idx):
        if self.sample_length:
            length = np.random.uniform(self.min_length,self.max_length)
        else:
            length = self.max_length

        inputs,targets = generate_dataset(self.batch_size,length,self.sample_length,self.trajectory_type)

        inputs = torch.tensor(inputs).float()
        targets = torch.tensor(targets).float()

        return inputs,targets