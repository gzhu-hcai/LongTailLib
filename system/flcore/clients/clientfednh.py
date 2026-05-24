"""
FedNH Client - Faithful reimplementation from FedNH-main source code.

Key design matching source:
- FedNHModel wrapper: backbone + prototype (frozen nn.Parameter) + scaling (learnable nn.Parameter)
- Prototype is NOT optimized during training (requires_grad=False)
- Scaling IS optimized during training (requires_grad=True)
- ALL parameters in model.state_dict() including prototype & scaling
- Stepwise LR: base_lr first half, base_lr*0.1 second half
- SGD(momentum=0.0, weight_decay=1e-5), gradient clipping max_norm=10
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data


class FedNHModel(nn.Module):
    """
    Wrapper matching ResNetModNH from FedNH-main/src/flbase/models/CNN.py.
    Wraps backbone with prototype + scaling as nn.Parameters.
    forward() returns logits (or (embedding, logits) when return_embedding=True).
    """

    def __init__(self, backbone, num_classes, embed_dim):
        super().__init__()
        self.backbone = backbone
        self.return_embedding = False

        # Replace classifier with Identity (backbone still returns (feature, identity(feature)))
        if hasattr(backbone, 'classifier'):
            backbone.classifier = nn.Identity()
        elif hasattr(backbone, 'fc'):
            backbone.fc = nn.Identity()

        # Prototype: (num_classes, embed_dim) matching source CNN.py line 105-106
        temp = nn.Linear(embed_dim, num_classes, bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)

        # Scaling: init 20.0 matching source CNN.py line 108
        self.scaling = nn.Parameter(torch.tensor([20.0]))

    def forward(self, x):
        out = self.backbone(x)
        if isinstance(out, tuple):
            feature_embedding = out[0]
        else:
            feature_embedding = out

        # L2 normalize embedding (source CNN.py line 113-114)
        feature_embedding_norm = torch.norm(
            feature_embedding, p=2, dim=1, keepdim=True
        ).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)

        # Normalize prototype conditionally (source CNN.py line 115-119)
        if self.prototype.requires_grad == False:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(
                self.prototype, p=2, dim=1, keepdim=True
            ).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)

        # Logits (source CNN.py line 120-121)
        logits = self.scaling * torch.matmul(feature_embedding, normalized_prototype.T)

        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits


class clientFedNH(Client):
    """
    FedNH Client following FedUHClient + FedNHClient from source.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # Count samples per class (source client.py line 43)
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        count_by_class = Counter()
        for _, y in train_data:
            if isinstance(y, torch.Tensor):
                y = y.item()
            count_by_class[int(y)] += 1
        self.count_by_class = dict(count_by_class)

        # count_by_class_full tensor (source FedNH.py line 26-27)
        temp = [self.count_by_class.get(cls, 1e-12) for cls in range(self.num_classes)]
        self.count_by_class_full = torch.tensor(temp).to(self.device)

        # Label distribution (source client.py line 44)
        total = sum(self.count_by_class.values())
        self.label_dist = {k: v / total for k, v in self.count_by_class.items()}

        # FedNH flags
        self.client_adv = getattr(args, 'FedNH_client_adv_prototype_agg', False)

        # Round tracking (set by server each round)
        self.current_round = 0
        self.total_rounds = getattr(args, 'global_rounds', 200)

        self.new_state_dict = None

    def train(self):
        """Local training matching FedUHClient.training() (FedUH.py line 79-120)."""
        trainloader = self.load_train_data()

        # Stepwise LR (source utils.py line 14-18)
        if self.current_round < self.total_rounds // 2:
            lr = self.learning_rate
        else:
            lr = self.learning_rate * 0.1

        # SGD on requires_grad params (source utils.py line 25-27, main.py line 206-207)
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, momentum=0.0, weight_decay=1e-5
        )

        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.loss(logits, y)

                # Backward (source FedUH.py line 105-108)
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=filter(lambda p: p.requires_grad, self.model.parameters()),
                    max_norm=10
                )
                optimizer.step()

        # Save post-training state_dict (source FedUH.py line 118)
        self.new_state_dict = self.model.state_dict()

    def upload(self):
        """Upload state_dict + estimated prototype (source FedNH.py line 95-99)."""
        if self.client_adv:
            return self.new_state_dict, self._estimate_prototype_adv()
        else:
            return self.new_state_dict, self._estimate_prototype()

    def _estimate_prototype(self):
        """Standard prototype estimation (source FedNH.py line 29-58)."""
        self.model.eval()
        self.model.return_embedding = True
        prototype = torch.zeros_like(self.model.prototype)

        # Use full training data without drop_last (matching source trainloader)
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        trainloader = DataLoader(train_data, self.batch_size, drop_last=False, shuffle=False)

        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                feature_embedding, _ = self.model(x)
                classes_shown = torch.unique(y).cpu().numpy()
                for cls in classes_shown:
                    mask = (y == cls)
                    prototype[int(cls)] += torch.sum(feature_embedding[mask, :], dim=0)

        for cls in self.count_by_class.keys():
            prototype[cls] /= self.count_by_class[cls]
            prototype_cls_norm = torch.norm(prototype[cls]).clamp(min=1e-12)
            prototype[cls] = torch.div(prototype[cls], prototype_cls_norm)
            prototype[cls] *= self.count_by_class[cls]

        self.model.return_embedding = False
        return {'scaled_prototype': prototype, 'count_by_class_full': self.count_by_class_full}

    def _estimate_prototype_adv(self):
        """Adversarial prototype estimation (source FedNH.py line 60-93)."""
        self.model.eval()
        self.model.return_embedding = True
        embeddings, labels_list, weights = [], [], []
        prototype = torch.zeros_like(self.model.prototype)

        train_data = read_client_data(self.dataset, self.id, is_train=True)
        trainloader = DataLoader(train_data, self.batch_size, drop_last=False, shuffle=False)

        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                feature_embedding, logits = self.model(x)
                prob_ = F.softmax(logits, dim=1)
                prob = torch.gather(prob_, dim=1, index=y.view(-1, 1))
                labels_list.append(y)
                weights.append(prob)
                embeddings.append(feature_embedding)

        self.model.return_embedding = False
        embeddings = torch.cat(embeddings, dim=0)
        labels_cat = torch.cat(labels_list, dim=0)
        weights = torch.cat(weights, dim=0).view(-1, 1)

        for cls in self.count_by_class.keys():
            mask = (labels_cat == cls)
            weights_in_cls = weights[mask, :]
            feature_embedding_in_cls = embeddings[mask, :]
            prototype[cls] = (
                torch.sum(feature_embedding_in_cls * weights_in_cls, dim=0)
                / torch.sum(weights_in_cls)
            )
            prototype_cls_norm = torch.norm(prototype[cls]).clamp(min=1e-12)
            prototype[cls] = torch.div(prototype[cls], prototype_cls_norm)

        return {'adv_agg_prototype': prototype, 'count_by_class_full': self.count_by_class_full}

    # test_metrics() and train_metrics() inherited from clientbase.
    # FedNHModel.forward() returns logits directly → base class handles correctly.
