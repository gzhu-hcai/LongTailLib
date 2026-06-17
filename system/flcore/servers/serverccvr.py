import time
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from flcore.clients.clientccvr import clientCCVR, Local
from flcore.servers.serverbase import Server


class VRDataset(torch.utils.data.Dataset):
    """Virtual representation dataset for classifier retraining"""
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ReTrainModel(nn.Module):
    """Simple linear classifier for VR retraining - source code line 49-53"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)


class Global(object):
    """
    Global server for CCVR - copied from source code fedavg/server.py
    Handles: model aggregation, global distribution calculation, VR retraining
    """
    
    def __init__(self, model, device, num_classes):
        self.global_model = model
        self.device = device
        self.num_classes = num_classes
        self.retrain_model = None

    def model_aggregate(self, clients_models, client_weight):
        """
        FedAvg aggregation - source code line 25-37
        """
        new_params = {}
        for client_id, client_params in clients_models.items():
            weight = client_weight[client_id]
            for name, param in client_params.items():
                if name not in new_params:
                    new_params[name] = param.clone() * weight
                else:
                    new_params[name] += param.clone() * weight
        
        self.global_model.load_state_dict(new_params)

    def cal_global_gd(self, c_mean, c_cov, c_length):
        """
        Calculate global mean and covariance - source code line 141-179
        Uses the formula from the CCVR paper (Eq. 3 and 4)
        """
        clients = list(c_mean.keys())
        num_classes = len(c_mean[clients[0]])

        g_mean = []
        g_cov = []

        for c in range(num_classes):
            # Total samples for class c across all clients
            n_c = sum([c_length[k][c] for k in clients])

            if n_c == 0:
                # No data for this class
                g_mean.append(None)
                g_cov.append(None)
                continue

            # Initialize with zeros
            mean_c = np.zeros_like(np.array(c_mean[clients[0]][0]))
            cov_ck = np.zeros_like(np.array(c_cov[clients[0]][0]))
            mul_mean = np.zeros_like(np.array(c_cov[clients[0]][0]))

            for k in clients:
                if c_length[k][c] > 0:
                    # Weighted mean - Eq. (3)
                    mean_c += (c_length[k][c] / n_c) * np.array(c_mean[k][c])

                    # For covariance calculation
                    mean_ck = np.array(c_mean[k][c])
                    if n_c > 1:
                        mul_mean += (c_length[k][c] / (n_c - 1)) * np.dot(mean_ck.reshape(-1, 1), mean_ck.reshape(1, -1))
                        cov_ck += ((c_length[k][c] - 1) / (n_c - 1)) * np.array(c_cov[k][c])

            g_mean.append(mean_c)

            # Global covariance - Eq. (4)
            if n_c > 1:
                cov_c = cov_ck + mul_mean - (n_c / (n_c - 1)) * np.dot(mean_c.reshape(-1, 1), mean_c.reshape(1, -1))
            else:
                cov_c = np.array(c_cov[clients[0]][c])

            g_cov.append(cov_c)

        # Convert to dict format for compatibility
        global_mean = {i: g_mean[i] for i in range(num_classes)}
        global_cov = {i: g_cov[i] for i in range(num_classes)}

        return global_mean, global_cov

    def retrain_vr(self, vr, label, eval_vr, eval_label, classifier, conf):
        """
        Retrain classifier using virtual representations - source code line 105-139
        """
        self.retrain_model = classifier
        retrain_dataset = VRDataset(vr, label)
        retrain_loader = DataLoader(retrain_dataset, batch_size=conf['batch_size'], shuffle=True)

        optimizer = torch.optim.SGD(
            self.retrain_model.parameters(), 
            lr=conf['retrain_lr'],
            momentum=conf['momentum'],
            weight_decay=conf['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()

        for e in range(conf['retrain_epoch']):
            self.retrain_model.train()

            for batch_id, batch in enumerate(retrain_loader):
                data, target = batch
                data = data.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                output = self.retrain_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            acc, eval_loss = self.model_eval_vr(eval_vr, eval_label, conf)
            print(f"  Retraining epoch {e} done. train_loss={loss.item():.4f}, eval_loss={eval_loss:.4f}, eval_acc={acc:.2f}%")

        return self.retrain_model

    @torch.no_grad()
    def model_eval_vr(self, eval_vr, label, conf):
        """Evaluate retrain model on VR - source code line 67-103"""
        self.retrain_model.eval()

        eval_dataset = VRDataset(eval_vr, label)
        eval_loader = DataLoader(eval_dataset, batch_size=conf['batch_size'], shuffle=False)

        total_loss = 0.0
        correct = 0
        dataset_size = 0

        criterion = nn.CrossEntropyLoss()
        for batch_id, batch in enumerate(eval_loader):
            data, target = batch
            data = data.to(self.device)
            target = target.to(self.device)
            dataset_size += data.size()[0]

            output = self.retrain_model(data)
            total_loss += criterion(output, target).item() * data.size()[0]
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        return acc, total_l


class FedCCVR(Server):
    """CCVR Server: Classifier Calibration with Virtual Representations"""
    
    def __init__(self, args, times):
        self._set_ccvr_defaults(args)
        super().__init__(args, times)
        
        # CCVR-specific parameters
        self.retrain_epoch = args.retrain_epoch
        self.retrain_lr = args.retrain_lr
        self.num_vr = args.num_vr
        self.momentum = getattr(args, 'momentum', 0.9)
        self.weight_decay = getattr(args, 'weight_decay', 1e-5)
        
        self.set_slow_clients()
        self.set_clients(clientCCVR)
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Local epochs per round: {self.local_epochs}")
        print("Finished creating server and clients.")
        
        # Initialize CCVR global handler
        self.ccvr_global = Global(
            model=copy.deepcopy(self.global_model),
            device=self.device,
            num_classes=self.num_classes
        )
        
        # Detect feature dimension
        self.feature_dim = self._detect_feature_dim()
        
        # Store client data references for CCVR phase 2
        self.list_client2data = {}
        for client in self.clients:
            # Get training data for each client
            train_data = client.load_train_data()
            # Store the dataset object
            self.list_client2data[client.id] = train_data.dataset
        
        self._print_config()

    def _print_config(self):
        print(f"\n{'='*60}")
        print(f"CCVR Configuration:")
        print(f"  num_clients: {self.num_clients}")
        print(f"  join_ratio: {self.join_ratio}")
        print(f"  global_rounds: {self.global_rounds}")
        print(f"  local_epochs: {self.local_epochs}")
        print(f"  retrain_epoch: {self.retrain_epoch}")
        print(f"  retrain_lr: {self.retrain_lr}")
        print(f"  num_vr: {self.num_vr}")
        print(f"  feature_dim: {self.feature_dim}")
        print(f"{'='*60}\n")

    def _set_ccvr_defaults(self, args):
        """Set CCVR default parameters"""
        import re
        if not hasattr(args, 'num_clients') or args.num_clients is None or args.num_clients == 0:
            match = re.search(r'NC(\d+)', args.dataset)
            args.num_clients = int(match.group(1)) if match else 20
        
        if not hasattr(args, 'retrain_epoch') or args.retrain_epoch is None:
            args.retrain_epoch = 10
        if not hasattr(args, 'retrain_lr') or args.retrain_lr is None:
            args.retrain_lr = 0.001
        if not hasattr(args, 'num_vr') or args.num_vr is None:
            args.num_vr = 2000

    def _detect_feature_dim(self):
        """Detect feature dimension from model output"""
        self.global_model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            try:
                output = self.global_model(dummy_input)
                if isinstance(output, tuple):
                    feature, _ = output
                    return feature.shape[1]
            except:
                pass
        return 256  # Default feature dimension

    def _get_conf(self):
        """Get configuration dict for source code compatibility"""
        return {
            'batch_size': self.batch_size,
            'retrain_epoch': self.retrain_epoch,
            'retrain_lr': self.retrain_lr,
            'num_vr': self.num_vr,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'num_classes': self.num_classes,
        }

    def train(self):
        """Main training loop - source code main.py"""
        
        # Phase 1: Standard FedAvg training
        print("\n=== Phase 1: FedAvg Training ===")
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            
            self.selected_clients = self.select_clients()
            self.send_models()
            
            # Local training
            clients_models = {}
            client_weight = {}
            total_samples = sum([c.train_samples for c in self.selected_clients])
            
            for client in self.selected_clients:
                client.train()
                clients_models[client.id] = copy.deepcopy(client.model.state_dict())
                client_weight[client.id] = client.train_samples / total_samples
            
            # FedAvg aggregation
            self.ccvr_global.model_aggregate(clients_models, client_weight)
            self.global_model.load_state_dict(self.ccvr_global.global_model.state_dict())
            
            self.Budget.append(time.time() - s_t)
            
            if i % self.eval_gap == 0:
                print(f"\n-------------Round {i}-------------")
                self.evaluate()
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])
            
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        
        # Phase 2: CCVR post-processing
        print("\n=== Phase 2: CCVR Calibration ===")
        self._ccvr_calibration()
        
        print("\n=== Training Complete ===")
        print(f"Average time cost per round: {sum(self.Budget)/len(self.Budget):.2f}s")
        
        self.save_results()
        self.save_global_model()

    def _ccvr_calibration(self):
        """CCVR calibration using virtual representations - source code line 76-130"""
        print("\nComputing local feature distributions...")
        
        client_mean = {}
        client_cov = {}
        client_length = {}
        
        # Each client computes local feature mean and covariance
        for client in self.clients:
            local = Local(
                data_client=self.list_client2data[client.id],
                model=copy.deepcopy(self.global_model),
                device=self.device,
                num_classes=self.num_classes,
                batch_size=self.batch_size
            )
            c_mean, c_cov, c_length = local.cal_distributions()
            client_mean[client.id] = c_mean
            client_cov[client.id] = c_cov
            client_length[client.id] = c_length
        
        print("Computing global mean and covariance...")
        g_mean, g_cov = self.ccvr_global.cal_global_gd(client_mean, client_cov, client_length)
        
        print("Generating virtual representations...")
        retrain_vr = []
        label = []
        eval_vr = []
        eval_label = []
        
        for i in range(self.num_classes):
            if g_mean[i] is None or g_cov[i] is None:
                continue
                
            mean = np.squeeze(np.array(g_mean[i]))
            # Handle potential numerical issues with covariance matrix
            cov = np.array(g_cov[i])
            # Ensure covariance matrix is positive semi-definite
            cov = cov + np.eye(cov.shape[0]) * 1e-6
            
            try:
                # Generate virtual representations from Gaussian distribution
                vr = np.random.multivariate_normal(mean, cov, self.num_vr)
                retrain_vr.extend(vr.tolist())
                label.extend([i] * self.num_vr)
                
                # Generate eval VR (same distribution, fewer samples)
                eval_samples = max(self.num_vr // 10, 100)
                e_vr = np.random.multivariate_normal(mean, cov, eval_samples)
                eval_vr.extend(e_vr.tolist())
                eval_label.extend([i] * eval_samples)
            except np.linalg.LinAlgError:
                print(f"  Warning: Could not generate VR for class {i} (covariance matrix issue)")
                continue
        
        print(f"Generated {len(retrain_vr)} virtual representations for retraining")
        
        # Create retrain model (simple linear classifier)
        retrain_model = ReTrainModel(self.feature_dim, self.num_classes).to(self.device)
        
        # Copy classifier weights from global model
        reset_name = []
        for name, _ in retrain_model.state_dict().items():
            reset_name.append(name)
        
        for name, param in self.global_model.state_dict().items():
            # Match classifier layer names
            short_name = name.split('.')[-1] if '.' in name else name
            for rn in reset_name:
                if short_name in rn or rn in name:
                    if retrain_model.state_dict()[rn].shape == param.shape:
                        retrain_model.state_dict()[rn].copy_(param.clone())
                        break
        
        print("Retraining classifier with virtual representations...")
        retrain_model = self.ccvr_global.retrain_vr(
            retrain_vr, label, eval_vr, eval_label, retrain_model, self._get_conf()
        )
        
        # Update global model classifier with retrained weights
        for name, param in retrain_model.state_dict().items():
            for gname, gparam in self.global_model.state_dict().items():
                short_gname = gname.split('.')[-1] if '.' in gname else gname
                if name.split('.')[-1] == short_gname and param.shape == gparam.shape:
                    self.global_model.state_dict()[gname].copy_(param.clone())
                    break
        
        print("\nEvaluation after CCVR calibration:")
        self.evaluate()
