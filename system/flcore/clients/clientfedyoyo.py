import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.optim import SGD
from torch.utils.data import DataLoader
from flcore.clients.clientbase import Client
from utils.data_utils import read_data, read_client_data, get_transforms
import torchvision.transforms as transforms


class clientFedYoYo(Client):
    """
    FedYoYo client implementation.
    Source: FedYoYo-master/main_fedyoyo.py class Local
    
    Key components:
    - Augmented Self-bootstrap Distillation (ASD): weak-aug teacher, strong-aug student
    - Distribution-aware Logit Adjustment (DLA): logit += log(prior^tau)
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # Override optimizer: SGD without momentum (source: main_fedyoyo.py line 154)
        self.optimizer = SGD(self.model.parameters(), lr=self.learning_rate)

        self.criterion = nn.CrossEntropyLoss()

        # FedYoYo hyperparameters (set by server)
        self.lamda = getattr(args, 'yoyo_lamda', 4.0)
        self.T = getattr(args, 'yoyo_T', 1.5)
        self.tau = getattr(args, 'yoyo_tau', 1.5)
        self.gamma = getattr(args, 'yoyo_gamma', 0.1)
        self.warmup = getattr(args, 'yoyo_warmup', 50)

        # Class distribution for this client
        self.cls_num_list = self._get_class_distribution()

        # Build augmentation transforms (source: main_fedyoyo.py lines 298-325)
        self._build_transforms()

    def _get_class_distribution(self):
        """Get per-class sample counts for this client (read raw data, no drop_last)."""
        data = read_data(self.dataset, self.id, is_train=True)
        class_counts = [0] * self.num_classes
        for label in data['y']:
            class_counts[int(label)] += 1
        return class_counts

    def _build_transforms(self):
        """Build weak/strong/no-aug transforms matching source code."""
        dataset_lower = self.dataset.lower()
        if 'cifar100' in dataset_lower:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif 'cifar10' in dataset_lower:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        else:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]

        normalize = transforms.Normalize(mean=mean, std=std)

        # No augmentation (source: test_trsfm, line 308-312)
        self.transform_none = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize,
        ])

        # Weak augmentation (source: augmentation_weak, line 301-307)
        self.transform_weak = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])

        # Strong augmentation (source: augmentation_strong, line 316-325)
        # AutoAugment + Cutout
        self.transform_strong = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            _Cutout(n_holes=1, length=16),
            normalize,
        ])

    def load_train_data_triple(self):
        """Load training data with triple augmentation (img0, img1, img2, label).
        Source: main_fedyoyo.py line 327 — dataset returns (img0, img1, img2, target, index)
        """
        data = read_data(self.dataset, self.id, is_train=True)
        dataset = TripleAugDataset(
            data['x'], data['y'],
            self.transform_none, self.transform_weak, self.transform_strong
        )
        return DataLoader(dataset, self.batch_size, drop_last=True, shuffle=True)

    def get_feature_mean(self, global_params):
        """Compute per-class feature mean using no-aug images.
        Source: main_fedyoyo.py Local.get_feature_mean() lines 198-220
        """
        self.model.load_state_dict(global_params)
        self.model.eval()
        out_dim = self.model.classifier.in_features
        feature_mean = torch.zeros(self.num_classes, out_dim).to(self.device)

        data = read_data(self.dataset, self.id, is_train=True)
        dataset = _SimpleAugDataset(data['x'], data['y'], self.transform_none)
        loader = DataLoader(dataset, self.batch_size, shuffle=True)

        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                features, _ = self.model(imgs)
                features = features.detach()
                for feat, label in zip(features, labels):
                    feature_mean[label] += feat

        cls_num_tensor = torch.tensor(self.cls_num_list).unsqueeze(1).to(self.device)
        for i in range(self.num_classes):
            if self.cls_num_list[i] > 0:
                feature_mean[i] = (feature_mean[i] / cls_num_tensor[i]).detach()
        return feature_mean

    def calculate_eff_weight(self, train_prototype):
        """Calculate effective weight per class based on feature dispersion.
        Source: main_fedyoyo.py Local.calculate_eff_weight() lines 222-264
        """
        EPS = 1e-6
        self.model.eval()
        train_prototype = train_prototype.to(self.device)
        eff_all = torch.zeros(self.num_classes).float().to(self.device)

        data = read_data(self.dataset, self.id, is_train=True)
        dataset = _SimpleAugDataset(data['x'], data['y'], self.transform_none)
        loader = DataLoader(dataset, self.batch_size, shuffle=True)

        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                features, _ = self.model(imgs)
                mu = train_prototype[labels].detach()
                feature_bz = features.detach() - mu  # Centralization
                index = torch.unique(labels)
                eff = torch.zeros(self.num_classes).float().to(self.device)

                for cls in index:
                    mask = (labels == cls).nonzero(as_tuple=False).squeeze()
                    feat_cls = feature_bz[mask].detach()

                    if feat_cls.dim() == 1:
                        eff[cls] = 1
                    else:
                        # Cosine similarity matrix
                        dot = torch.matmul(feat_cls, feat_cls.t())
                        norms = torch.sqrt(torch.sum(feat_cls ** 2, dim=1)).unsqueeze(1)
                        norm_prod = torch.matmul(norms, norms.t())
                        norm_prod[norm_prod == 0] = EPS
                        r = dot / norm_prod
                        num = feat_cls.size(0)
                        a = torch.ones(1, num, device=self.device) / num
                        b = torch.ones(num, 1, device=self.device) / num
                        c = torch.matmul(torch.matmul(a, r), b).float()
                        if c < EPS:
                            c = torch.tensor(EPS, device=self.device)
                        eff[cls] = 1.0 / c

                eff_all += eff
        return eff_all

    def train_with_prior(self, prior, current_round):
        """Local training with ASD + DLA.
        Source: main_fedyoyo.py Local.local_train() lines 156-195
        """
        self.model.train()
        start_time = time.time()
        local_dis = torch.tensor(self.cls_num_list, dtype=torch.float, device=self.device)
        local_dis /= (local_dis.sum() + 1e-9)
        # prior is mutated progressively across batches (source: line 174)
        prior = prior.clone().to(self.device)

        for _ in range(self.local_epochs):
            loader = self.load_train_data_triple()
            for img0, img1, img2, labels in loader:
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

                # Concatenate weak + strong (source: line 170-171)
                data = torch.cat([img1, img2], dim=0)
                target = torch.cat([labels, labels], dim=0)

                _, logits = self.model(data)

                # DLA: Distribution-aware Logit Adjustment (source: line 174-175)
                prior = (1 - self.gamma) * prior + self.gamma * local_dis
                logits = logits + torch.log(torch.pow(prior, self.tau) + 1e-9)

                num = target.shape[0] // 2

                # ASD: Augmented Self-bootstrap Distillation (source: line 177-187)
                teacher_logits = logits[:num, :]
                student_logits = logits[num:, :]
                teacher_softmax = F.softmax(teacher_logits / self.T, dim=1).detach()
                student_logsoftmax = F.log_softmax(student_logits / self.T, dim=1)
                teacher_max, teacher_index = torch.max(
                    F.softmax(teacher_logits, dim=1).detach(), dim=1
                )
                partial_target = target[:num]

                # Only distill on correctly predicted samples (source: line 183-184)
                correct_mask = (teacher_index == partial_target)
                if correct_mask.sum() > 0:
                    kd_loss = F.kl_div(
                        student_logsoftmax[correct_mask],
                        teacher_softmax[correct_mask],
                        reduction='batchmean'
                    )
                    if torch.isnan(kd_loss):
                        kd_loss = 0
                else:
                    kd_loss = 0

                # CE loss on all (source: line 188)
                ce_loss = self.criterion(logits, target)

                # Total loss with warmup (source: line 189)
                loss = ce_loss + self.lamda * kd_loss * min(current_round / self.warmup, 1.0)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        return self.model.state_dict()


class TripleAugDataset:
    """Dataset returning (img_no_aug, img_weak, img_strong, label)."""
    def __init__(self, X, y, t_none, t_weak, t_strong):
        self.X = X
        self.y = y
        self.t_none = t_none
        self.t_weak = t_weak
        self.t_strong = t_strong

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if isinstance(x, np.ndarray):
            if x.ndim == 3 and (x.shape[0] == 3 or x.shape[0] == 1):
                x = x.transpose(1, 2, 0)
            if x.dtype != np.uint8:
                x = (x * 255).astype(np.uint8) if x.dtype in [np.float32, np.float64] else x.astype(np.uint8)
        img0 = self.t_none(x)
        img1 = self.t_weak(x)
        img2 = self.t_strong(x)
        return img0, img1, img2, torch.tensor(y, dtype=torch.long)


class _SimpleAugDataset:
    """Dataset with single transform for feature extraction."""
    def __init__(self, X, y, transform):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if isinstance(x, np.ndarray):
            if x.ndim == 3 and (x.shape[0] == 3 or x.shape[0] == 1):
                x = x.transpose(1, 2, 0)
            if x.dtype != np.uint8:
                x = (x * 255).astype(np.uint8) if x.dtype in [np.float32, np.float64] else x.astype(np.uint8)
        return self.transform(x), torch.tensor(y, dtype=torch.long)


class _Cutout:
    """Cutout augmentation (source: data_loader/autoaug.py Cutout class)."""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        # img: Tensor (C, H, W)
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones(h, w, dtype=img.dtype)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0
        return img * mask.unsqueeze(0)
