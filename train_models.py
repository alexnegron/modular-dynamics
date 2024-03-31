#this is a script for training and evaluating a single model on a single task
#it is intended to be run from the command line
#this way we can do a grid sweep of hyperparameters for the model
#and easily assess scaling properties wrt model and task complexity

import torch
from src.models import RNNBase
from tasks.task_dataloader import IntegrationDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import hydra
import os
import matplotlib.pyplot as plt
import h5py
import logging
import pickle
from torch.nn.functional import cosine_similarity

logger = logging.getLogger(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_tensor_type(torch.DoubleTensor)

def angular_loss(outputs,targets):
    output_angle = torch.arctan(outputs[:,1]/outputs[:,0])
    target_angle = torch.arctan(targets[:,1]/targets[:,0])
    loss = torch.mean(torch.sqrt(1-(output_angle-target_angle)))
    return loss

def initialize_loss_fxn(training_cfg):
    if training_cfg.loss == 'mse':
        criterion = torch.nn.MSELoss()
    elif training_cfg.loss == 'angular':
        criterion = angular_loss
    elif training_cfg.loss == 'cosine':
        criterion = lambda x1,x2: cosine_similarity(x1,x2,dim=-1).mean()
    
    return criterion

def train_model(model,dataloader,training_cfg):

    criterion = initialize_loss_fxn(training_cfg)

    optimizer = optim.Adam(model.parameters(),lr=training_cfg.lr,weight_decay=training_cfg.weight_decay)

    losses = []
    clip_value = training_cfg.clip_value
    model.to(device)
    for epoch in range(training_cfg.epochs):

        for i,data in enumerate(dataloader):

            inputs,targets = data

            inputs = inputs.to(device).to(torch.float64)
            targets = targets.to(device).to(torch.float64)
            print(inputs.shape,targets.shape)
            optimizer.zero_grad()
            _, outputs = model(inputs)

            loss = criterion(outputs,targets)
            
            loss.backward()

            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip_value)

            optimizer.step()
            losses.append(loss.item())

            logger.info(f'Epoch {epoch} Loss: {loss.item()}')

    return losses

def initialize_dataloader(cfg):
    task = IntegrationDataset(cfg.task.dim,cfg.task.batch_size,cfg.task.min_length,
                              cfg.task.max_length,cfg.task.sample_length)
    
    dataloader = DataLoader(task,batch_size=cfg.task.batch_size,shuffle=False)

    return task,dataloader

def test_model(cfg,model,task):
    
    criterion = initialize_loss_fxn(cfg.training)

    task.set_test()
    task.batch_size=cfg.task.test_batch_size
    dataloader = DataLoader(task,batch_size=cfg.task.test_batch_size,shuffle=False)

    net_loss = 0.0
    all_inputs,all_outputs,all_targets = [],[],[]
    for i,data in enumerate(dataloader):

        inputs,targets = data
        inputs = inputs.to(device).to(torch.float64)
        targets = targets.to(device).to(torch.float64)
        print(inputs.shape,targets.shape)

        hiddens, outputs = model(inputs)

        loss = criterion(outputs,targets)
        net_loss += loss
        logger.info(f'Test Loss: {loss.item()}')

    net_loss /= len(dataloader)
    all_inputs = inputs.cpu().numpy()
    all_outputs = outputs.cpu().detach().numpy()
    all_targets = targets.cpu().numpy()
    all_hidden_activity = hiddens.detach().cpu().numpy()
    
    return net_loss.item(),all_hidden_activity,all_inputs,all_outputs,all_targets

def save_results(model,losses,net_loss,hidden_activity,all_inputs,all_outputs,all_targets):
    #same model to pickle
    with open('model.pkl','wb') as f:
        pickle.dump(model,f)

    #save the rest to h5py
    with h5py.File('results.h5','w') as f:
        f.create_dataset('losses',data=losses)
        f.create_dataset('net_loss',data=net_loss)
        f.create_dataset('hidden_activity',data=hidden_activity)
        f.create_dataset('all_inputs',data=all_inputs)
        f.create_dataset('all_outputs',data=all_outputs)
        f.create_dataset('all_targets',data=all_targets)

def plot_results(cfg,results):

    pass

@hydra.main(config_path='conf',config_name='config')
def main(cfg):
    #TODO: handling noise
    rnn_kwargs = {} if cfg.model.rnn_kwargs is None else dict(cfg.model.rnn_kwargs)

    model = RNNBase(cfg.model.architecture,cfg.model.neurons,cfg.model.activation,cfg.task.dim,
                2* cfg.task.dim,cfg.model.seed,**rnn_kwargs)
    
    model.to(device)
    
    task,dataloader = initialize_dataloader(cfg)

    losses = train_model(model,dataloader,cfg.training)

    net_loss,hidden_activity,all_inputs,all_outputs,all_targets = test_model(cfg,model,task)

    save_results(model,losses,net_loss,hidden_activity,all_inputs,all_outputs,all_targets)

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()


