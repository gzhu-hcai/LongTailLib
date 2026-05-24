<<<<<<< HEAD
"""
FedNH Server - Faithful reimplementation from FedNH-main source code.

Key design matching source:
- FedNHModel wrapper: prototype + scaling inside model's state_dict
- exclude_layer_keys = EMPTY (source YAML: exclude: [])
  → ALL keys aggregated via FedAvg (prototype unchanged since frozen; scaling averaged)
  → Prototype then overwritten via moving average of client-estimated prototypes
- Server LR = 1.0, LR decay = 1.0 (standard FedAvg averaging)
- Smoothing = 0.9 for prototype moving average
- Evaluation uses parent serverbase.evaluate()
"""

import copy
import time
import torch
import torch.nn.functional as F
from flcore.clients.clientfednh import clientFedNH, FedNHModel
from flcore.servers.serverbase import Server


class FedNH(Server):
    def __init__(self, args, times):
        # ─── FedNH default parameters (source YAML + main.py defaults) ───
        # Client training defaults
        args.local_learning_rate = getattr(args, 'local_learning_rate', 0.1)  # client_lr
        args.local_epochs = getattr(args, 'local_epochs', 5)  # num_epochs
        args.batch_size = getattr(args, 'batch_size', 64)

        # Server defaults
        args.num_clients = getattr(args, 'num_clients', 100)
        args.join_ratio = getattr(args, 'join_ratio', 0.1)  # participate_ratio

        # FedNH-specific defaults (source main.py line 221-223, YAML line 36-37)
        self.smoothing = getattr(args, 'FedNH_smoothing', 0.9)
        self.server_adv = getattr(args, 'FedNH_server_adv_prototype_agg', False)
        self.client_adv = getattr(args, 'FedNH_client_adv_prototype_agg', False)
        self.fix_scaling = getattr(args, 'FedNH_fix_scaling', False)

        # Server learning rate (source YAML line 27-28)
        self.server_lr = getattr(args, 'server_lr', 1.0)
        self.server_lr_decay = getattr(args, 'server_lr_decay', 1.0)

        # Store on args for client access
        args.FedNH_client_adv_prototype_agg = self.client_adv

        # ─── Create FedNHModel wrapper from backbone ───
        backbone = copy.deepcopy(args.model)

        # Infer embedding dimension from backbone's classifier
        if hasattr(backbone, 'classifier') and hasattr(backbone.classifier, 'in_features'):
            embed_dim = backbone.classifier.in_features
        elif hasattr(backbone, 'fc') and hasattr(backbone.fc, 'in_features'):
            embed_dim = backbone.fc.in_features
        else:
            embed_dim = 512  # default for ResNet18

        wrapped = FedNHModel(backbone, args.num_classes, embed_dim)

        # Orthogonal init for prototype (source FedUH.py line 59-60)
        m, n = wrapped.prototype.shape
        wrapped.prototype.data = torch.nn.init.orthogonal_(torch.rand(m, n))
        wrapped.prototype.requires_grad_(False)

        # Fix scaling if requested (source FedUH.py line 73-76)
        if self.fix_scaling:
            wrapped.scaling.requires_grad_(False)
            wrapped.scaling.data = torch.tensor([30.0])

        # Move to correct device
        wrapped = wrapped.to(args.device)

        # Replace args.model so super().__init__ and clients use the wrapper
        args.model = wrapped

        # ─── Initialize base server ───
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientFedNH)

        # Server model state_dict (ALL keys: backbone + prototype + scaling)
        self.server_model_state_dict = copy.deepcopy(self.clients[0].model.state_dict())

        # NO exclude layer keys (source YAML: exclude: [])
        # Prototype effectively unchanged by FedAvg since it's frozen
        self.exclude_layer_keys = set()

        self.embed_dim = embed_dim

        print(f"[FedNH] smoothing={self.smoothing}, server_adv={self.server_adv}, "
              f"client_adv={self.client_adv}, fix_scaling={self.fix_scaling}")
        print(f"[FedNH] server_lr={self.server_lr}, embed_dim={self.embed_dim}")
        print(f"[FedNH] prototype shape: {wrapped.prototype.shape}, "
              f"scaling init: {wrapped.scaling.item():.1f}")

    def send_models(self):
        """Send full server_model_state_dict to selected clients (no exclusions)."""
        for client in self.selected_clients:
            # load_state_dict preserves requires_grad flags
            client.model.load_state_dict(self.server_model_state_dict)
            client.current_round = self._current_round
            client.total_rounds = self.global_rounds

    def receive_and_aggregate(self):
        """
        Two-phase aggregation matching source FedNHServer.aggregate():
        1. FedAvg on ALL keys (no exclusions)
        2. Overwrite prototype with moving-average of client-estimated prototypes
        """
        # Collect uploads: (state_dict, prototype_dict) per client
        client_uploads = []
        for client in self.selected_clients:
            sd, proto_dict = client.upload()
            client_uploads.append((sd, proto_dict))

        num_participants = len(client_uploads)
        server_lr = self.server_lr * (self.server_lr_decay ** (self._current_round - 1))

        with torch.no_grad():
            # ── Phase 1: FedAvg aggregation on ALL keys ──
            # Accumulate sum of (client - server) updates
            update_direction = {}
            # Also collect prototype aggregation info
            cumsum_per_class = torch.zeros(self.num_classes, device=self.device)
            agg_weights_dict = {}

            for idx, (client_state, proto_dict) in enumerate(client_uploads):
                # Prototype aggregation weights
                if not self.server_adv:
                    cumsum_per_class += proto_dict['count_by_class_full'].to(self.device)
                else:
                    mu = proto_dict['adv_agg_prototype'].to(self.device)
                    W = self.server_model_state_dict['prototype']
                    agg_weights_dict[idx] = torch.exp(
                        torch.sum(W * mu, dim=1, keepdim=True)
                    )

                # FedAvg: accumulate client_state - server_state
                for key in self.server_model_state_dict:
                    c_val = client_state[key].to(self.device)
                    s_val = self.server_model_state_dict[key].to(self.device)
                    delta = c_val.float() - s_val.float()
                    if idx == 0:
                        update_direction[key] = delta.clone()
                    else:
                        update_direction[key] += delta

            # Apply FedAvg: server = server + (server_lr / N) * sum_updates
            for key in self.server_model_state_dict:
                self.server_model_state_dict[key] = (
                    self.server_model_state_dict[key].to(self.device).float()
                    + (server_lr / num_participants) * update_direction[key]
                ).to(self.server_model_state_dict[key].dtype)

            # ── Phase 2: Overwrite prototype with moving average ──
            avg_prototype = torch.zeros_like(self.server_model_state_dict['prototype'])

            if not self.server_adv:
                # Standard: weighted average by class counts (source FedNH.py line 147-148)
                for _, proto_dict in client_uploads:
                    avg_prototype += (
                        proto_dict['scaled_prototype'].to(self.device)
                        / cumsum_per_class.view(-1, 1)
                    )
            else:
                # Adversarial (source FedNH.py line 150-155)
                m = self.server_model_state_dict['prototype'].shape[0]
                sum_of_weights = torch.zeros((m, 1), device=self.device)
                for idx, (_, proto_dict) in enumerate(client_uploads):
                    sum_of_weights += agg_weights_dict[idx]
                    avg_prototype += (
                        agg_weights_dict[idx]
                        * proto_dict['adv_agg_prototype'].to(self.device)
                    )
                avg_prototype /= sum_of_weights

            # Normalize (source FedNH.py line 158)
            avg_prototype = F.normalize(avg_prototype, dim=1)

            # Moving average update (source FedNH.py line 160-164)
            weight = self.smoothing
            temp = weight * self.server_model_state_dict['prototype'] + (1 - weight) * avg_prototype
            self.server_model_state_dict['prototype'] = F.normalize(temp, dim=1)

    def evaluate(self, acc=None, loss=None):
        """Sync global_model before parent evaluation."""
        self.global_model.load_state_dict(self.server_model_state_dict)
        self.global_model.to(self.device)
        super().evaluate(acc, loss)

    def train(self):
        """Main FL training loop."""
        for i in range(1, self.global_rounds + 1):
            s_t = time.time()
            self._current_round = i
            self.selected_clients = self.select_clients()

            # Send global model to selected clients
            self.send_models()

            # Client local training
            for client in self.selected_clients:
                client.train()

            # Aggregate
            self.receive_and_aggregate()

            self.Budget.append(time.time() - s_t)
            print(f"\n----- Round {i}/{self.global_rounds} -----  "
                  f"Time: {self.Budget[-1]:.2f}s")

            # Evaluate
            if i % self.eval_gap == 0:
                self.evaluate()

        # Final save
        print("\nTraining complete.")
        print(f"Average time per round: {sum(self.Budget) / len(self.Budget):.2f}s")
        self.save_results()
        self.save_global_model()
=======
import copy
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
from sklearn import metrics

from flcore.servers.serverbase import Server
from flcore.clients.clientfednh import clientFedNH


class FedNHNative(Server):
    """FedNH server with prototype aggregation and extended metrics."""

    @staticmethod
    def register_cli_aliases(parser):
        """Register FedNH-specific command-line arguments (following FedNH-main)"""
        # Core FL parameters
        parser.add_argument('--num_clients_fednh', type=int, default=100,
                            help="FedNH: number of clients (default: 100)")
        parser.add_argument('--join_ratio_fednh', type=float, default=0.1,
                            help="FedNH: client participation ratio (default: 0.1)")
        
        # FedNH-specific parameters
        parser.add_argument('--FedNH_smoothing', type=float, default=0.9,
                            help="FedNH: moving average parameter for prototype update (default: 0.9)")
        parser.add_argument('--FedNH_server_adv_prototype_agg', action='store_true',
                            help="FedNH: use adversarial prototype aggregation on server (default: False)")
        parser.add_argument('--FedNH_client_adv_prototype_agg', action='store_true',
                            help="FedNH: use adversarial prototype aggregation on client (default: False)")
        parser.add_argument('--FedNH_scale', type=float, default=20.0,
                            help="FedNH: scaling parameter for prototype logits (default: 20.0, matching FedNH-main)")

    def __init__(self, args, times):
        # Override random seed for FedNH
        args.seed = 42
        
        # Set FedNH default parameters (following FedNH-main/experiments/Cifar10_Conv2Cifar.yaml)
        # These can be overridden by command-line arguments if needed
        args.num_clients = 100
        args.join_ratio = 0.1  # participate_ratio in FedNH-main
        
        # FedNH-specific hyperparameters (read from args, with defaults matching FedNH-main)
        self.fednh_smoothing = getattr(args, 'FedNH_smoothing', 0.9)
        self.fednh_server_adv = getattr(args, 'FedNH_server_adv_prototype_agg', False)
        self.fednh_client_adv = getattr(args, 'FedNH_client_adv_prototype_agg', False)
        self.fednh_scale = getattr(args, 'FedNH_scale', 20.0)  # FedNH-main default: 20.0
        
        # Pass FedNH-specific params to args for client initialization
        args.FedNH_smoothing = self.fednh_smoothing
        args.FedNH_server_adv_prototype_agg = self.fednh_server_adv
        args.FedNH_client_adv_prototype_agg = self.fednh_client_adv
        args.FedNH_scale = self.fednh_scale
        
        super().__init__(args, times)

        # Set slow clients and use FedNH client type
        self.set_slow_clients()
        self.set_clients(clientFedNH)

        # Build global prototype (following FedNH-main FedUH.py lines 59-60: orthogonal init)
        d = self._infer_feature_dim(self.global_model)
        self.embed_dim = d
        proto_init = torch.nn.init.orthogonal_(torch.rand(self.num_classes, d))
        self.global_prototype = nn.Parameter(proto_init.clone().to(self.device), requires_grad=False)
        
        # Debug: print FedNH hyperparameters
        print(f"[FedNH] scaling={self.fednh_scale}, smoothing={self.fednh_smoothing}")

        # counters for extended evaluation
        self._eval_counter = 0
        
        # FedNH paper metrics: GM, PM(V), PM(L)
        self.rs_GM_acc = []  # Global Model accuracy (uniform)
        self.rs_PM_V_acc = []  # Personalized Model accuracy (validclass)
        self.rs_PM_L_acc = []  # Personalized Model accuracy (labeldist)
        
        # Best accuracy trackers
        self.best_GM_acc = 0.0
        self.best_PM_V_acc = 0.0
        self.best_PM_L_acc = 0.0

    def _infer_feature_dim(self, model: nn.Module) -> int:
        if hasattr(model, 'fc') and isinstance(getattr(model, 'fc'), nn.Linear):
            return int(model.fc.in_features)
        if hasattr(model, 'fc1'):
            fc1 = getattr(model, 'fc1')
            if isinstance(fc1, nn.Sequential):
                for layer in fc1:
                    if isinstance(layer, nn.Linear):
                        return int(layer.out_features)
            if isinstance(fc1, nn.Linear):
                return int(fc1.out_features)
        return 512

    # -------------------- sending --------------------
    def send_models(self):
        global_model = copy.deepcopy(self.global_model)
        for client in self.selected_clients:
            client.set_parameters(global_model)
            client.set_prototype(self.global_prototype.data.clone().detach())

    # -------------------- aggregation --------------------
    @torch.no_grad()
    def aggregate_parameters(self):
        """Aggregate client models and prototype statistics, update global model & prototype."""
        # Use state_dict to handle both parameters and buffers (e.g., BatchNorm running_mean/var)
        if hasattr(self, 'uploaded_models') and len(self.uploaded_models) > 0:
            # Initialize global model state dict to zeros ON SERVER DEVICE
            global_state = self.global_model.state_dict()
            for key in global_state.keys():
                global_state[key] = torch.zeros_like(global_state[key], device=self.device)
            
            # Aggregate using state_dict (handles both params and buffers)
            for mdl, w in zip(self.uploaded_models, self.uploaded_weights):
                client_state = mdl.state_dict()
                for key in global_state.keys():
                    # Skip non-floating point tensors (e.g., num_batches_tracked in BatchNorm)
                    if client_state[key].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                        global_state[key] = client_state[key].to(self.device)
                    else:
                        global_state[key] += client_state[key].to(self.device) * float(w)
            
            self.global_model.load_state_dict(global_state)
        elif len(self.selected_clients) > 0:
            tot_samples = sum(c.train_samples for c in self.selected_clients)
            global_state = self.global_model.state_dict()
            for key in global_state.keys():
                global_state[key] = torch.zeros_like(global_state[key], device=self.device)
            
            for c in self.selected_clients:
                w = float(c.train_samples) / float(max(tot_samples, 1))
                client_state = c.model.state_dict()
                for key in global_state.keys():
                    # Skip non-floating point tensors (e.g., num_batches_tracked in BatchNorm)
                    if client_state[key].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                        global_state[key] = client_state[key].to(self.device)
                    else:
                        global_state[key] += client_state[key].to(self.device) * w
            
            self.global_model.load_state_dict(global_state)

        # aggregate prototypes (following FedNH-main lines 112-164)
        d = self.embed_dim
        num_classes = self.num_classes
        
        if hasattr(self, 'uploaded_pkgs') and len(self.uploaded_pkgs) > 0:
            pkgs = self.uploaded_pkgs
        else:
            pkgs = [c.get_upload_package() for c in self.selected_clients]
        
        # Compute cumulative count per class for standard aggregation (FedNH-main line 113-118)
        cumsum_per_class = torch.zeros(num_classes).to(self.device)
        if not self.fednh_server_adv:
            for pkg in pkgs:
                # Tensors already on correct device from client
                cumsum_per_class += pkg['count_by_class_full']
        
        # Aggregate prototypes
        avg_prototype = torch.zeros(num_classes, d, device=self.device)
        
        if not self.fednh_server_adv:
            # Standard aggregation (FedNH-main lines 146-148)
            for pkg in pkgs:
                # Tensors already on correct device from client
                scaled_proto = pkg['scaled_prototype']
                # Divide by cumulative count (weighted average)
                avg_prototype += scaled_proto / cumsum_per_class.view(-1, 1)
        else:
            # Adversarial aggregation (FedNH-main lines 150-155)
            agg_weights_dict = {}
            for idx, pkg in enumerate(pkgs):
                # Tensors already on correct device from client
                mu = pkg['adv_agg_prototype']
                W = self.global_prototype.data
                # Compute attention weights
                agg_weights_dict[idx] = torch.exp(torch.sum(W * mu, dim=1, keepdim=True))
            
            sum_of_weights = torch.zeros((num_classes, 1)).to(self.device)
            for idx, pkg in enumerate(pkgs):
                sum_of_weights += agg_weights_dict[idx]
                avg_prototype += agg_weights_dict[idx] * pkg['adv_agg_prototype']
            avg_prototype /= sum_of_weights
        
        # Normalize prototype (line 158)
        avg_prototype = F.normalize(avg_prototype, dim=1)
        
        # Update prototype with moving average (lines 160-164)
        weight = self.fednh_smoothing
        temp = weight * self.global_prototype.data + (1 - weight) * avg_prototype
        # Normalize again
        self.global_prototype.data = F.normalize(temp, dim=1)

    # -------------------- extended evaluation --------------------
    @torch.no_grad()
    def evaluate(self, acc=None, loss=None):
        """FedNH evaluation with three paper metrics: GM, PM(V), PM(L)"""
        stats_train = self.train_metrics()
        # Weighted average train loss
        train_loss = sum([loss * ns for loss, ns in zip(stats_train[2], stats_train[1])]) / sum(stats_train[1])
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print(f"\n{'='*60}")
        print(f"Round {self.current_round}/{self.global_rounds}")
        print(f"{'='*60}")
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        
        # Compute GM (Global Model) - using global prototype on all clients' test sets
        GM_acc = self.compute_GM_accuracy()
        
        # Compute PM(V) and PM(L) - personalized models with different criteria
        PM_V_acc, PM_L_acc = self.compute_PM_accuracies()
        
        # Display metrics
        print("GM (Global Model):              {:.4f}".format(GM_acc))
        print("PM(V) (Personalized-ValidClass): {:.4f}".format(PM_V_acc))
        print("PM(L) (Personalized-LabelDist):  {:.4f}".format(PM_L_acc))
        
        # Store metrics
        self.rs_GM_acc.append(GM_acc)
        self.rs_PM_V_acc.append(PM_V_acc)
        self.rs_PM_L_acc.append(PM_L_acc)
        
        # Update best metrics
        self.best_GM_acc = max(self.best_GM_acc, GM_acc)
        self.best_PM_V_acc = max(self.best_PM_V_acc, PM_V_acc)
        self.best_PM_L_acc = max(self.best_PM_L_acc, PM_L_acc)
        
        # Maintain backward compatibility for rs_test_acc and rs_global_acc
        if not hasattr(self, 'rs_test_acc'):
            self.rs_test_acc = []
        if not hasattr(self, 'rs_global_acc'):
            self.rs_global_acc = []
        self.rs_test_acc.append(PM_V_acc)  # Use PM(V) as local accuracy
        self.rs_global_acc.append(GM_acc)  # Use GM as global accuracy

    @torch.no_grad()
    def compute_GM_accuracy(self):
        """Compute GM (Global Model) accuracy using global prototype on all clients' test sets."""
        try:
            total_correct = 0
            total_count = 0
            
            for c in self.clients:
                testloader = c.load_test_data()
                c.model.to(self.device)
                c.model.eval()
                
                for x, y in testloader:
                    if isinstance(x, list):
                        x_in = x[0].to(self.device)
                    else:
                        x_in = x.to(self.device)
                    y = y.to(self.device)
                    
                    # Extract embeddings and compute logits using global prototype
                    emb = c._forward_to_embedding(x_in)
                    emb_norm = F.normalize(emb, p=2, dim=1)
                    proto_norm = F.normalize(self.global_prototype.data, p=2, dim=1)
                    logits = c.fednh_scale * torch.matmul(emb_norm, proto_norm.t())
                    
                    pred = torch.argmax(logits, dim=1)
                    total_correct += (pred == y).sum().item()
                    total_count += y.size(0)
                
                c.model.cpu()
            
            if total_count > 0:
                return total_correct / total_count
            else:
                return 0.0
        except Exception as e:
            print(f"Warning: failed to compute GM accuracy: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    @torch.no_grad()
    def compute_PM_accuracies(self):
        """Compute PM(V) and PM(L) accuracies using personalized models with different criteria."""
        try:
            # Collect personalized test results from all clients
            total_PM_V_acc = 0.0
            total_PM_L_acc = 0.0
            num_clients = len(self.clients)
            
            for c in self.clients:
                # Use client's personalized test method with three criteria
                acc_by_criteria, _, _ = c.test_metrics_personalized()
                total_PM_V_acc += acc_by_criteria['validclass']
                total_PM_L_acc += acc_by_criteria['labeldist']
            
            # Average across all clients (FedNH-main FedAvg.py line 225-226)
            PM_V_acc = total_PM_V_acc / num_clients if num_clients > 0 else 0.0
            PM_L_acc = total_PM_L_acc / num_clients if num_clients > 0 else 0.0
            
            return PM_V_acc, PM_L_acc
        except Exception as e:
            print(f"Warning: failed to compute PM accuracies: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0

    # -------------------- receiving --------------------
    def receive_models(self):
        """Receive local models and prototype packages from active clients with drop emulation."""
        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients)
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_pkgs = []

        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = (
                    client.train_time_cost['total_cost'] / max(client.train_time_cost['num_rounds'], 1) +
                    client.send_time_cost['total_cost'] / max(client.send_time_cost['num_rounds'], 1)
                )
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                pkg = client.get_upload_package()
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                # Store client model directly (will move to device during aggregation)
                self.uploaded_models.append(client.model)
                self.uploaded_pkgs.append(pkg)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / max(tot_samples, 1)

    # -------------------- training loop --------------------
    def train(self):
        self.Budget = []
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.current_round = i
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                _ = self.evaluate()

            # local training
            for client in self.selected_clients:
                client.train()

            # receive weighted models and prototype packages, then aggregate
            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # Final results summary
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        # Display FedNH paper metrics: GM, PM(V), PM(L)
        if len(self.rs_GM_acc) > 0:
            print(f"Best GM (Global Model):              {self.best_GM_acc:.4f}")
        if len(self.rs_PM_V_acc) > 0:
            print(f"Best PM(V) (Personalized-ValidClass): {self.best_PM_V_acc:.4f}")
        if len(self.rs_PM_L_acc) > 0:
            print(f"Best PM(L) (Personalized-LabelDist):  {self.best_PM_L_acc:.4f}")
        
        if len(self.Budget) > 1:
            avg_time = sum(self.Budget[1:]) / len(self.Budget[1:])
            print(f"Average time per round: {avg_time:.2f}s")
        print(f"{'='*60}")

        self.save_results()
        self.save_global_model()

    def save_results(self):
        """Save FedNH results with GM, PM(V), PM(L) metrics"""
        import h5py
        import matplotlib.pyplot as plt
        import os
        import time
        
        try:
            # Create output directory
            ts = time.strftime("%Y%m%d_%H%M%S")
            base_name = f"{self.dataset}_{self.algorithm}_{ts}"
            base_dir = os.path.dirname(os.path.abspath(__file__))
            result_root = os.path.normpath(os.path.join(base_dir, '..', '..', '..', 'results'))
            run_dir = os.path.join(result_root, base_name)
            os.makedirs(run_dir, exist_ok=True)
            
            # Save metrics to HDF5
            if len(self.rs_GM_acc) > 0:
                file_path = os.path.join(run_dir, f"{base_name}.h5")
                print("File path: " + file_path)
                
                with h5py.File(file_path, 'w') as hf:
                    hf.create_dataset('rs_GM_acc', data=self.rs_GM_acc)
                    hf.create_dataset('rs_PM_V_acc', data=self.rs_PM_V_acc)
                    hf.create_dataset('rs_PM_L_acc', data=self.rs_PM_L_acc)
                    hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                    # For backward compatibility
                    hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                    hf.create_dataset('rs_global_acc', data=self.rs_global_acc)
                
                # Plot GM curve
                self._plot_metric_curve(self.rs_GM_acc, 'GM (Global Model)', 
                                       run_dir, base_name, '_GM', 'tab:blue')
                
                # Plot PM(V) curve
                self._plot_metric_curve(self.rs_PM_V_acc, 'PM(V) (Personalized-ValidClass)', 
                                       run_dir, base_name, '_PM_V', 'tab:green')
                
                # Plot PM(L) curve
                self._plot_metric_curve(self.rs_PM_L_acc, 'PM(L) (Personalized-LabelDist)', 
                                       run_dir, base_name, '_PM_L', 'tab:orange')
                
                # Plot combined comparison
                self._plot_combined_curves(run_dir, base_name)
                
        except Exception as e:
            print(f"Warning: save_results failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_metric_curve(self, data, label, run_dir, base_name, suffix, color):
        """Plot a single metric curve"""
        import matplotlib.pyplot as plt
        
        try:
            rounds = list(range(len(data)))
            plt.figure(figsize=(7, 4), dpi=150)
            plt.plot(rounds, data, marker='o', linewidth=1.8, markersize=3, color=color, label=label)
            plt.xlabel('Round')
            plt.ylabel('Test Accuracy')
            plt.title(f'{self.dataset}-{self.algorithm}: {label} per Round')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            if len(rounds) <= 20:
                plt.xticks(rounds)
            else:
                step = max(1, len(rounds) // 20)
                plt.xticks(list(range(0, rounds[-1] + 1, step)))
            
            plt.tight_layout()
            svg_path = os.path.join(run_dir, f"{base_name}{suffix}.svg")
            plt.savefig(svg_path, format='svg')
            plt.close()
            print(f"Saved {label} curve: {svg_path}")
        except Exception as e:
            print(f"Warning: failed to save {label} curve: {e}")
    
    def _plot_combined_curves(self, run_dir, base_name):
        """Plot all three metrics in one figure for comparison"""
        import matplotlib.pyplot as plt
        
        try:
            rounds = list(range(len(self.rs_GM_acc)))
            plt.figure(figsize=(9, 5), dpi=150)
            plt.plot(rounds, self.rs_GM_acc, marker='o', linewidth=1.8, markersize=3, 
                    color='tab:blue', label='GM (Global Model)')
            plt.plot(rounds, self.rs_PM_V_acc, marker='s', linewidth=1.8, markersize=3, 
                    color='tab:green', label='PM(V) (ValidClass)')
            plt.plot(rounds, self.rs_PM_L_acc, marker='^', linewidth=1.8, markersize=3, 
                    color='tab:orange', label='PM(L) (LabelDist)')
            plt.xlabel('Round')
            plt.ylabel('Test Accuracy')
            plt.title(f'{self.dataset}-{self.algorithm}: FedNH Metrics Comparison')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            if len(rounds) <= 20:
                plt.xticks(rounds)
            else:
                step = max(1, len(rounds) // 20)
                plt.xticks(list(range(0, rounds[-1] + 1, step)))
            
            plt.tight_layout()
            svg_path = os.path.join(run_dir, f"{base_name}_comparison.svg")
            plt.savefig(svg_path, format='svg')
            plt.close()
            print(f"Saved comparison curve: {svg_path}")
        except Exception as e:
            print(f"Warning: failed to save comparison curve: {e}")
>>>>>>> 15b6b60dba275c21157ead9a494232b7bb315b8d
