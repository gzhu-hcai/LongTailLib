
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data


class ClientGRAB(Client):
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # Source: options.py line 11-12
        self.lr = args.local_learning_rate  # 0.03
        self.momentum = 0.5  # options.py line 12
        
        # PIDLOSS and classifier will be set by server
        self.pid_loss = None
        self.local_classifier = None
        
        # Optimizer - Source: update_baseline.py line 98-100
        # ONLY optimizes backbone (net.parameters()), NOT classifier
        self.backbone_optimizer = torch.optim.SGD(
            self.model.base.parameters(),
            lr=self.lr,
            momentum=self.momentum
        )
    
    def set_pid_loss(self, pid_loss):
        """Set PIDLOSS from server"""
        self.pid_loss = pid_loss
    
    def set_classifier(self, classifier):
        """Set classifier from server"""
        self.local_classifier = classifier
    
    def train(self, g_backbone):
        # Load global backbone - Source: fed_grab.py line 94
        # net = copy.deepcopy(g_backbone)
        self.model.base.load_state_dict(copy.deepcopy(g_backbone.state_dict()))
        
        # Source: update_baseline.py line 94-95
        self.model.base.train()
        self.local_classifier.train()
        
        # Re-initialize optimizer for backbone + classifier
        self.backbone_optimizer = torch.optim.SGD(
            list(self.model.base.parameters()) + list(self.local_classifier.parameters()),
            lr=self.lr,
            momentum=self.momentum
        )
        
        # Get training data
        trainloader = self.load_train_data()
        
        # Training loop - Source: update_baseline.py line 103-125
        for epoch in range(self.local_epochs):
            for batch_idx, (images, labels) in enumerate(trainloader):
                # Source: update_baseline.py line 108-110
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                
                self.backbone_optimizer.zero_grad()
                
                # Source: update_baseline.py line 116-117
                feat = self.model.base(images)
                # Handle models that return (feature, logit) tuple
                if isinstance(feat, tuple):
                    feat = feat[0]
                logits = self.local_classifier(feat)
                
                # Source: update_baseline.py line 118
                if self.pid_loss is not None:
                    loss = self.pid_loss(logits, labels)
                else:
                    loss = nn.CrossEntropyLoss()(logits, labels)
                
                # Source: update_baseline.py line 119-120
                loss.backward()
                self.backbone_optimizer.step()
        
        # Source: update_baseline.py line 127
        # return net.state_dict(), GBA_Layer.state_dict(), loss
        return self.model.base.state_dict(), self.local_classifier.state_dict()
    
    @torch.no_grad()
    def test_metrics(self):
        self.model.base.eval()
        if self.local_classifier is not None:
            self.local_classifier.eval()
        
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        
        num_correct = 0
        num_samples = 0
        total_loss = 0.0
        
        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device).long()

            feat = self.model.base(images)
            # Handle models that return (feature, logit) tuple
            if isinstance(feat, tuple):
                feat = feat[0]
            if self.local_classifier is not None:
                outputs = self.local_classifier(feat)
            else:
                outputs = self.model.head(feat)
            
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            
            _, predicted = torch.max(outputs, 1)
            num_correct += (predicted == labels).sum().item()
            num_samples += labels.size(0)
        
        return {
            'num_samples': num_samples,
            'num_correct': num_correct,
            'loss': total_loss / num_samples if num_samples > 0 else 0.0
        }
