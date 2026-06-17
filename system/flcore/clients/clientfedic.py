

import copy
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data


class ClientFEDIC(Client):

    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # FedIC specific parameters - source: options.py line 28
        # lr_local_training = 0.1, SGD without momentum
        self.lr = getattr(args, 'local_learning_rate', 0.1)
        
        # Override optimizer: SGD without momentum (source: main.py line 283)
        self.optimizer = SGD(self.model.parameters(), lr=self.lr)
        
        # Loss function
        self.ce_loss = CrossEntropyLoss()
    
    def set_parameters_dict(self, dict_params):

        self.model.load_state_dict(dict_params)
    
    def train(self):

        # Safety: skip training if received model already has NaN params
        for p in self.model.parameters():
            if not torch.isfinite(p).all():
                print(f"[Client {self.id}] WARNING: model has NaN/Inf params, skipping training")
                return

        self.model.train()

        for epoch in range(self.local_epochs):
            # Create dataloader for each epoch (like source)
            train_data = read_client_data(self.dataset, self.id, is_train=True)
            data_loader = DataLoader(
                dataset=train_data,
                batch_size=self.batch_size,
                shuffle=True
            )

            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward - model returns (feature, logit)
                _, outputs = self.model(images)

                # Loss
                loss = self.ce_loss(outputs, labels)

                if not torch.isfinite(loss):
                    continue

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
    
    def upload_params(self):

        return copy.deepcopy(self.model.state_dict())
    
    # test_metrics() and train_metrics() inherited from clientbase
    # clientbase already handles (feature, logit) tuple at line 120-121
