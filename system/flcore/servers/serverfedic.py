"""
FedIC: Federated Learning with Inter-Client Distillation
Direct copy from FEDIC-main/main.py

Reference:
    FEDIC-main/main.py - Global class (server), Local class (client), Ensemble_highway
    FEDIC-main/options.py - default parameters
    
Key features:
- Three-stage server-side training:
  1. FedAvg refinement (100 steps)
  2. Highway ensemble training (100 steps)
  3. Knowledge distillation to global model (100 steps)
- Ensemble_highway: weighted ensemble of client logits + FedAvg logits
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import sigmoid, cat
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax, log_softmax
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
from torchvision import datasets as tv_datasets
from torchvision import transforms as tv_transforms

from flcore.clients.clientfedic import ClientFEDIC
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from flcore.trainmodel.resnet_cifar import resnet8_cifar, resnet18_cifar, resnet20_cifar


class Ensemble_highway(nn.Module):
    """
    Highway ensemble module - Direct copy from FEDIC-main/main.py line 23-64
    
    Computes weighted ensemble of client logits with highway gating.
    """
    def __init__(self, num_classes=10, feature_dim=256, num_online_clients=8):
        super(Ensemble_highway, self).__init__()
        # calibration
        self.ensemble_scale = nn.Parameter(torch.ones(num_classes, 1))
        self.ensemble_bias = nn.Parameter(torch.zeros(1))

        self.logit_scale = nn.Parameter(torch.ones(num_classes))
        self.logit_bias = nn.Parameter(torch.zeros(num_classes))
        self.classifier2 = nn.Linear(in_features=feature_dim, out_features=1)
        self.carry_values = []
        self.weight_values = []
        self.num_online_clients = num_online_clients
        
    def forward(self, step, clients_feature, clients_logit, new_logit):
        """
        Direct copy from source: main.py line 35-64
        """
        all_logits_weight = torch.mm(clients_logit[0], self.ensemble_scale)
        all_logits_weight = all_logits_weight + self.ensemble_bias
        all_logits_weight_sigmoid = sigmoid(all_logits_weight)
        for one_logit in clients_logit[1:]:
            new_value = torch.mm(one_logit, self.ensemble_scale)
            new_value = new_value + self.ensemble_bias
            new_value_sigmoid = sigmoid(new_value)
            all_logits_weight_sigmoid = cat((all_logits_weight_sigmoid, new_value_sigmoid), dim=1)
        norm1 = all_logits_weight_sigmoid.norm(1, dim=1)
        norm1 = norm1.unsqueeze(1).expand_as(all_logits_weight_sigmoid)
        all_logits_weight_norm = all_logits_weight_sigmoid / norm1
        all_logits_weight_norm = all_logits_weight_norm.t()
        weighted_logits = sum([
            one_weight.view(-1, 1) * one_logit
            for one_logit, one_weight in zip(clients_logit, all_logits_weight_norm)
        ])
        # Average feature weights - source uses 1/8 for 8 clients
        num_clients = len(clients_feature)
        avg_weight = [1.0 / num_clients] * num_clients
        weighted_feature = sum([
            one_weight * one_feature
            for one_feature, one_weight in zip(clients_feature, avg_weight)
        ])
        calibration_logit = weighted_logits * self.logit_scale + self.logit_bias
        carry_gate = self.classifier2(weighted_feature)
        carry_gate_sigmoid = sigmoid(carry_gate)
        finally_logit = carry_gate_sigmoid * calibration_logit + (1 - carry_gate_sigmoid) * new_logit
        return finally_logit


class FedIC(Server):
    """
    FedIC Server - Direct copy from FEDIC-main/main.py Global class
    
    Three-stage server-side training per round:
    1. FedAvg refinement: Train model2 with labeled data (100 steps)
    2. Highway ensemble: Train highway_model with client logits (100 steps)
    3. Distillation: Distill ensemble knowledge to global model (100 steps)
    """
    
    @staticmethod
    def register_cli_aliases(parser):
        """Register FedIC-specific arguments - from options.py"""
        # Local training
        parser.add_argument('--fedic_local_epochs', type=int, default=10,
                            help='FedIC: local epochs (source: 10)')
        parser.add_argument('--fedic_batch_size', type=int, default=128,
                            help='FedIC: local batch size (source: 128)')
        parser.add_argument('--fedic_lr_local', type=float, default=0.1,
                            help='FedIC: local learning rate (source: 0.1)')
        
        # Server-side distillation
        parser.add_argument('--fedic_lr_global', type=float, default=0.001,
                            help='FedIC: server learning rate (source: 0.001)')
        parser.add_argument('--fedic_temperature', type=float, default=2.0,
                            help='FedIC: distillation temperature (source: 2)')
        parser.add_argument('--fedic_ld', type=float, default=0.5,
                            help='FedIC: soft/hard loss ratio (source: 0.5)')
        parser.add_argument('--fedic_mini_batch_size', type=int, default=20,
                            help='FedIC: labeled batch size for server (source: 20)')
        parser.add_argument('--fedic_mini_batch_size_unlabeled', type=int, default=128,
                            help='FedIC: unlabeled batch size for server (source: 128)')
        
        # Client selection
        parser.add_argument('--fedic_num_clients', type=int, default=20,
                            help='FedIC: number of clients (source: 20)')
        parser.add_argument('--fedic_num_online_clients', type=int, default=8,
                            help='FedIC: online clients per round (source: 8)')
    
    def __init__(self, args, times):
        # Override with FedIC source code defaults BEFORE parent init
        # Source: options.py
        
        # Fixed seed from source code - line 35: seed=7
        args.random_seed = getattr(args, 'random_seed', 7)
        
        # num_clients - source: line 11: num_clients=20
        args.num_clients = getattr(args, 'fedic_num_clients', 20)
        
        # local_epochs - source: line 17: num_epochs_local_training=10
        args.local_epochs = getattr(args, 'fedic_local_epochs', 10)
        
        # batch_size - source: line 18: batch_size_local_training=128
        args.batch_size = getattr(args, 'fedic_batch_size', 128)
        
        # learning rate - source: line 28: lr_local_training=0.1
        args.local_learning_rate = getattr(args, 'fedic_lr_local', 0.1)

        # num_online_clients - source: line 12: num_online_clients=8
        num_online_clients = getattr(args, 'fedic_num_online_clients', 8)
        args.join_ratio = num_online_clients / args.num_clients

        # Support flexible model selection
        dataset_name = args.dataset.lower()
        model_type = getattr(args, 'model', 'ResNet8')
        if isinstance(model_type, str) and 'cifar' in dataset_name:
            if model_type in ['ResNet18', 'resnet18']:
                print(f"\n[FedIC] Using resnet18_cifar (feature_dim=512)")
                args.model = resnet18_cifar(num_classes=args.num_classes).to(args.device)
                self.feature_dim = 512
            elif model_type in ['ResNet20', 'resnet20']:
                print(f"\n[FedIC] Using resnet20_cifar (feature_dim=256)")
                args.model = resnet20_cifar(num_classes=args.num_classes).to(args.device)
                self.feature_dim = 256
            else:  # Default: ResNet8
                print(f"\n[FedIC] Using resnet8_cifar (feature_dim=256)")
                args.model = resnet8_cifar(num_classes=args.num_classes, scaling=4).to(args.device)
                self.feature_dim = 256
        else:
            self.feature_dim = 256  # Default

        super().__init__(args, times)

        # FedIC hyperparameters - source: options.py
        self.lr_global_teaching = getattr(args, 'fedic_lr_global', 0.001)
        self.temperature = getattr(args, 'fedic_temperature', 2.0)
        self.ld = getattr(args, 'fedic_ld', 0.5)
        self.mini_batch_size = getattr(args, 'fedic_mini_batch_size', 20)
        self.mini_batch_size_unlabeled = getattr(args, 'fedic_mini_batch_size_unlabeled', 128)
        self.batch_size_test = getattr(args, 'batch_size_test', 500)
        self.num_online_clients = num_online_clients

        # Initialize models - source: main.py line 83-96
        # model: global model for distillation output
        # model1: for computing client features/logits
        # model2: FedAvg refined model
        self.model1 = copy.deepcopy(self.global_model).to(self.device)
        self.model2 = copy.deepcopy(self.global_model).to(self.device)

        # Highway ensemble model - source: main.py line 95
        self.highway_model = Ensemble_highway(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            num_online_clients=self.num_online_clients
        ).to(self.device)

        # Global params dict - source: main.py line 97
        self.dict_global_params = self.global_model.state_dict()

        # Optimizers - source: main.py line 104-106
        self.optimizer = Adam(self.global_model.parameters(), lr=self.lr_global_teaching, weight_decay=0.0002)
        self.highway_optimizer = Adam(self.highway_model.parameters(), lr=self.lr_global_teaching)
        self.fedavg_optimizer = Adam(self.model2.parameters(), lr=self.lr_global_teaching, weight_decay=0.0002)
        
        # Loss function
        self.ce_loss = CrossEntropyLoss()
        
        # Random state - source: main.py line 119
        self.random_state = np.random.RandomState(args.random_seed)
        
        # Load global teaching dataset (labeled data for server-side training)
        self._load_teaching_data()
        
        # Create clients
        self.set_slow_clients()
        self.set_clients(ClientFEDIC)
        
        # Tracking
        self.epoch_acc = []
        
        print(f"\n{'='*60}")
        print(f"FedIC Configuration (source code defaults):")
        print(f"  seed: {args.random_seed} (source: 7)")
        print(f"  num_clients: {self.num_clients} (source: 20)")
        print(f"  num_online_clients: {self.num_online_clients} (source: 8)")
        print(f"  join_ratio: {self.join_ratio:.2f}")
        print(f"  global_rounds: {self.global_rounds}")
        print(f"  local_epochs: {self.local_epochs} (source: 10)")
        print(f"  batch_size: {self.batch_size} (source: 128)")
        print(f"  lr_local: {self.learning_rate} (source: 0.1)")
        print(f"  lr_global: {self.lr_global_teaching} (source: 0.001)")
        print(f"  temperature: {self.temperature} (source: 2)")
        print(f"  ld (soft/hard ratio): {self.ld} (source: 0.5)")
        print(f"  mini_batch_size: {self.mini_batch_size} (source: 20)")
        print(f"  mini_batch_size_unlabeled: {self.mini_batch_size_unlabeled} (source: 128)")
        print(f"  feature_dim: {self.feature_dim}")
        print(f"{'='*60}\n")
    
    def _load_teaching_data(self):
        """
        Load teaching and unlabeled data from ORIGINAL BALANCED CIFAR-10.
        
        Source data flow (main.py line 310-335):
        - transform_train: RandomCrop(32,4) + RandomHorizontalFlip + Normalize
        - partition_train_teach: train=indices[:4900], teach=indices[4900:] (100/class, 1000 total)
        - unlabeled_data = full CIFAR-10 train set (50000 balanced, with augmentation)
        - teaching_data = 1000 balanced samples (100/class, with augmentation)
        
        Critical: both use on-the-fly augmentation via transform_train.
        """
        rawdata_path = os.path.join('../dataset', self.dataset, 'rawdata')
        
        # Source transform_train (main.py line 310-315)
        transform_train = tv_transforms.Compose([
            tv_transforms.RandomCrop(32, padding=4),
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Load full original CIFAR-10 training set (50000, balanced, with augmentation)
        full_cifar10 = tv_datasets.CIFAR10(
            rawdata_path, train=True, download=True, transform=transform_train
        )
        
        # Unlabeled data = full training set (source: main.py line 322)
        self.unlabeled_data = full_cifar10
        
        # Teaching data: partition_train_teach (source: dataset.py line 23-32)
        # num_data_train=49000, per class: train=indices[:4900], teach=indices[4900:]
        # seed = args.seed = 7
        random_state_data = np.random.RandomState(7)  # source: RandomState(args.seed=7)
        targets = np.array(full_cifar10.targets)
        teach_indices = []
        
        for cls in range(self.num_classes):
            cls_indices = np.where(targets == cls)[0].tolist()
            random_state_data.shuffle(cls_indices)
            num_train_per_class = 49000 // self.num_classes  # 4900
            teach_indices.extend(cls_indices[num_train_per_class:])  # 100 per class = teach
        
        self.teaching_data = Subset(full_cifar10, teach_indices)
        
        print(f"[FedIC] Teaching data: {len(self.teaching_data)} balanced samples from original CIFAR-10")
        print(f"[FedIC] Unlabeled data: {len(self.unlabeled_data)} samples (full balanced CIFAR-10)")
    
    def _get_teaching_batch(self, batch_size, use_unlabeled=False):
        """
        Sample batch with on-the-fly augmentation (RandomCrop + RandomHorizontalFlip).
        Each call to dataset[idx] applies fresh random augmentation.
        Source: main.py line 133-145, 181-186
        """
        dataset = self.unlabeled_data if use_unlabeled else self.teaching_data
        total = len(dataset)
        batch_indices = self.random_state.choice(total, batch_size, replace=False)
        
        images = []
        labels = []
        for idx in batch_indices:
            image, label = dataset[idx]
            images.append(image)
            labels.append(torch.tensor(label))
        
        images = torch.stack(images).to(self.device)
        labels = torch.stack(labels).to(self.device)
        return images, labels
    
    def _features_logits(self, images, list_dicts_local_params):
        """
        Compute features and logits from all client models.
        Source: main.py line 225-235
        """
        list_features = []
        list_logits = []
        
        for dict_local_params in list_dicts_local_params:
            self.model1.load_state_dict(dict_local_params)
            self.model1.eval()
            with torch.no_grad():
                local_feature, local_logits = self.model1(images)
                list_features.append(copy.deepcopy(local_feature))
                list_logits.append(copy.deepcopy(local_logits))
        
        return list_features, list_logits
    
    def _initialize_for_model_fusion(self, list_dicts_local_params, list_nums_local_data):
        """
        FedAvg aggregation of client params.
        Source: main.py line 237-243
        """
        for name_param in self.dict_global_params:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            self.dict_global_params[name_param] = value_global_param
    
    def update_distillation_highway_feature(self, round_idx, list_dicts_local_params, list_nums_local_data):
        """
        Three-stage server update - Direct copy from source: main.py line 127-223
        
        Stage 1: FedAvg refinement (100 steps)
        Stage 2: Highway ensemble training (100 steps)
        Stage 3: Knowledge distillation to global model (100 steps)
        """
        # Initialize global params with FedAvg
        self._initialize_for_model_fusion(copy.deepcopy(list_dicts_local_params), list_nums_local_data)
        
        # ========== Stage 1: FedAvg refinement (100 steps) ==========
        # Source: main.py line 130-150
        self.model2.load_state_dict(self.dict_global_params)
        self.model2.train()
        
        for hard_step in tqdm(range(100), desc='Stage 1: FedAvg refinement'):
            images, labels = self._get_teaching_batch(self.mini_batch_size)
            _, fedavg_outputs = self.model2(images)
            fedavg_hard_loss = self.ce_loss(fedavg_outputs, labels)
            self.fedavg_optimizer.zero_grad()
            fedavg_hard_loss.backward()
            self.fedavg_optimizer.step()
        
        self.model2.eval()
        
        # ========== Stage 2: Highway ensemble training (100 steps) ==========
        # Source: main.py line 152-176
        self.highway_model.train()
        
        for ensemble_step in tqdm(range(100), desc='Stage 2: Highway ensemble'):
            images, labels = self._get_teaching_batch(self.mini_batch_size)
            
            ensemble_feature_temp, ensemble_logit_temp = self._features_logits(
                images, copy.deepcopy(list_dicts_local_params)
            )
            
            # Source line 170: no no_grad() wrapper on model2 (eval mode, only highway_optimizer steps)
            _, fedavg_new_logits = self.model2(images)
            
            ensemble_avg_logit_finally = self.highway_model(
                ensemble_step, ensemble_feature_temp, ensemble_logit_temp, fedavg_new_logits
            )
            
            ensemble_hard_loss = self.ce_loss(ensemble_avg_logit_finally, labels)
            self.highway_optimizer.zero_grad()
            ensemble_hard_loss.backward()
            self.highway_optimizer.step()
        
        self.highway_model.eval()
        
        # ========== Stage 3: Distillation to global model (100 steps) ==========
        # Source: main.py line 178-218
        self.global_model.load_state_dict(self.dict_global_params)
        self.global_model.train()
        
        for step in tqdm(range(100), desc='Stage 3: Distillation'):
            # Get unlabeled batch
            images_unlabeled, _ = self._get_teaching_batch(self.mini_batch_size_unlabeled, use_unlabeled=True)
            
            # Get labeled batch
            images_labeled, labels_train = self._get_teaching_batch(self.mini_batch_size)
            
            # Teacher: source computes without no_grad(), uses y.detach() at KL loss
            # _features_logits already uses no_grad() internally (source: main.py line 231)
            teacher_feature_temp, teacher_logits_temp = self._features_logits(
                images_unlabeled, copy.deepcopy(list_dicts_local_params)
            )
            _, fedavg_unlabeled_logits = self.model2(images_unlabeled)
            logits_teacher = self.highway_model(
                round_idx, teacher_feature_temp, teacher_logits_temp, fedavg_unlabeled_logits
            )
            
            # Student: global model on unlabeled data
            _, logits_student = self.global_model(images_unlabeled)
            
            # Soft loss: KL divergence
            x = log_softmax(logits_student / self.temperature, dim=1)
            y = softmax(logits_teacher / self.temperature, dim=1)
            soft_loss = F.kl_div(x, y.detach(), reduction='batchmean')
            
            # Hard loss: CE on labeled data
            _, logits_student_train = self.global_model(images_labeled)
            hard_loss = self.ce_loss(logits_student_train, labels_train)
            
            # Total loss - source: main.py line 214
            total_loss = self.ld * soft_loss + (1 - self.ld) * hard_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        # Update global params
        self.dict_global_params = self.global_model.state_dict()
    
    def evaluate(self, acc=None, loss=None):
        """Sync global_model and all clients before parent evaluation."""
        self.global_model.load_state_dict(self.dict_global_params)
        self.global_model.to(self.device)
        # Sync all clients so parent's test_metrics() works correctly
        for client in self.clients:
            client.set_parameters(self.global_model)
        super().evaluate(acc, loss)

    def train(self):
        """Main training loop - Source: main.py line 353-381"""
        for round_idx in range(1, self.global_rounds + 1):
            s_t = time.time()
            
            # Select online clients
            self.selected_clients = self.select_clients()
            
            # Send global params to selected clients (source: main.py line 354)
            dict_global_params = copy.deepcopy(self.dict_global_params)
            for client in self.selected_clients:
                client.set_parameters_dict(dict_global_params)
            
            # Local training (source: main.py line 360-374)
            list_dicts_local_params = []
            list_nums_local_data = []
            
            for client in self.selected_clients:
                client.train()
                dict_local_params = client.upload_params()
                list_dicts_local_params.append(dict_local_params)
                list_nums_local_data.append(client.train_samples)
            
            # Server-side update (three-stage distillation)
            self.update_distillation_highway_feature(
                round_idx,
                copy.deepcopy(list_dicts_local_params),
                list_nums_local_data
            )
            
            self.Budget.append(time.time() - s_t)
            print(f"\n----- Round {round_idx}/{self.global_rounds} -----  "
                  f"Time: {self.Budget[-1]:.2f}s")
            
            # Evaluate using parent class
            if round_idx % self.eval_gap == 0:
                self.evaluate()
        
        print("\nFedIC Training Complete")
        print(f"Average time per round: {sum(self.Budget) / len(self.Budget):.2f}s")
        self.save_results()
        self.save_global_model()
