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
