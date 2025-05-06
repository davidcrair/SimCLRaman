"""Contrastive trainer for pretraining spectral encoder
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from loss_functions import InfoNCELoss
from loss_functions import SupConLoss

class ContrastiveTrainer:
    def __init__(
        self, 
        model, 
        dataloader, 
        device, 
        loss_fn_name="supcon", 
        temperature=0.05, 
        lr=3e-4
    ):
        """
        initialize the contrastive trainer
        
        Args:
            model: the SpectralNet model to train
            dataloader: DataLoader for   contrastive training data
            device: device to train on
            loss_fn_name: loss function to use ('supcon' or 'infonce')
            temperature: temperature param. for contrastive loss
            lr: learning rate for optimizer
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.lr = lr
        
        # set up optimizer to train encoder and projector
        self.optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.linear_projector.parameters()),
            lr=self.lr
        )
        
        # set up loss function
        if loss_fn_name.lower() == "infonce":
            self.loss_fn = InfoNCELoss(temperature=temperature)
        else:  # dfault to supervised contrastive
            self.loss_fn = SupConLoss(temperature=temperature)
        
        # initialize tracking variables
        self.loss_history = []
        self.acc_history = []
    
    def train(self, epochs=10):
        """train model for the specified number of epochs"""
        for epoch in range(epochs + 1):
            self.model.train()
            epoch_loss = 0
            epoch_acc = 0
            num_batches = 0
            
            for (x_views, y) in self.dataloader:
                # x_views contains the two augmented views for each input
                # each view is a batch of spectra
                x_view1, x_view2 = x_views
                x_view1, x_view2 = x_view1.to(self.device), x_view2.to(self.device)
                
                # get projections for both views
                z1 = self.model.project(x_view1)
                z2 = self.model.project(x_view2)
                loss = self.loss_fn(z1, z2, labels=y.to(self.device))
                        
                # back prop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # epoch stats
            epoch_loss /= num_batches
            self.loss_history.append(epoch_loss)
            self.acc_history.append(epoch_acc)
            
            print(f'Contrastive Epoch: {epoch} Loss: {epoch_loss:.4f}')
        
        return self.loss_history, self.acc_history
    
    def save_model(self, save_dir='saved_models', prefix='contrastive'):
        """Save the model and encoder weights to the specified directory."""
        # create directory to save models if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # determine epochs from length of loss history
        epochs = len(self.loss_history) - 1
        
        # save entire model
        model_save_path = os.path.join(save_dir, f'{prefix}_model_e{epochs}_cnn.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'acc_history': self.acc_history,
            'epochs': epochs
        }, model_save_path)
        
        # save only the encoder weights
        encoder_save_path = os.path.join(save_dir, f'{prefix}_encoder_e{epochs}_cnn.pt')
        torch.save({
            'encoder_state_dict': self.model.encoder.state_dict(),
            'epochs': epochs
        }, encoder_save_path)
        
        print(f"Model saved to {model_save_path}")
        print(f"Encoder saved to {encoder_save_path}")
        
        return model_save_path, encoder_save_path