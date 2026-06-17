
import time
import copy
import numpy as np
import torch

from flcore.servers.serverbase import Server
from flcore.clients.clientfedyoyo import clientFedYoYo


def apply_change_threshold(cur_eff_all, previous_eff_all, change_threshold):
    """Clamp outlier eff_weight changes.
    Source: FedYoYo-master/data_loader/tools.py apply_change_threshold()
    """
    epsilon = 1e-8
    change_ratio = torch.abs((cur_eff_all - previous_eff_all) / (previous_eff_all + epsilon))
    exceed_threshold_mask = change_ratio > change_threshold
    cur_eff_all[exceed_threshold_mask] = previous_eff_all[exceed_threshold_mask]
    return cur_eff_all.clone()


class FedYoYo(Server):
    """
    FedYoYo server.
    Two-phase per round:
      Phase 1: Compute effective weights (feature dispersion) → prior
      Phase 2: Local training with ASD + DLA, then FedAvg aggregation
    """

    # ---- Default hyperparameters (aligned with fedyoyo.sh + argparse) ----
    DEFAULT_LAMDA = 4.0        # KD loss weight (fedyoyo.sh)
    DEFAULT_T = 1.5            # Distillation temperature (argparse default)
    DEFAULT_TAU = 1.5          # Logit adjustment exponent (argparse default)
    DEFAULT_GAMMA = 0.1        # Prior blend coefficient (fedyoyo.sh)
    DEFAULT_WARMUP = 50        # KD warmup rounds (fedyoyo.sh)

    def __init__(self, args, times):
        # Inject FedYoYo defaults into args before base __init__
        args.yoyo_lamda = getattr(args, 'yoyo_lamda', self.DEFAULT_LAMDA)
        args.yoyo_T = getattr(args, 'yoyo_T', self.DEFAULT_T)
        args.yoyo_tau = getattr(args, 'yoyo_tau', self.DEFAULT_TAU)
        args.yoyo_gamma = getattr(args, 'yoyo_gamma', self.DEFAULT_GAMMA)
        args.yoyo_warmup = getattr(args, 'yoyo_warmup', self.DEFAULT_WARMUP)

        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedYoYo)

        # Global effective weight state (source: line 388)
        self.eff_global_pre = torch.zeros(self.num_classes, dtype=torch.float, device=self.device)

        print(f"\n[FedYoYo] lamda={args.yoyo_lamda}, T={args.yoyo_T}, "
              f"tau={args.yoyo_tau}, gamma={args.yoyo_gamma}, warmup={args.yoyo_warmup}")
        print(f"[FedYoYo] Join ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for r in range(self.global_rounds + 1):
            s_t = time.time()

            self.selected_clients = self.select_clients()

            # Distribute global model to ALL clients (source: line 390)
            self.send_models()

            # Evaluate before training
            if r % self.eval_gap == 0:
                print(f"\n-------------Round number: {r}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            # ===== Phase 1: Compute effective weights =====
            global_params = self.global_model.state_dict()
            eff_global_cur = torch.ones(self.num_classes, dtype=torch.float, device=self.device)

            for client in self.selected_clients:
                # Feature prototype extraction (source: line 401)
                local_feature_mean = client.get_feature_mean(global_params)
                local_feature_mean = local_feature_mean.detach()
                # Effective weight computation (source: line 403)
                eff_all = client.calculate_eff_weight(local_feature_mean)
                eff_global_cur = eff_global_cur + eff_all

            # EMA smoothing + outlier clamping (source: lines 406-412)
            if r > 1:
                eff_global_cur = self.eff_global_pre * 0.9 + eff_global_cur * 0.1
                eff_global_cur = apply_change_threshold(eff_global_cur, self.eff_global_pre, 100)
            else:
                abnormal = eff_global_cur > 1e+4
                if (~abnormal).any():
                    mean_val = eff_global_cur[~abnormal].mean()
                    eff_global_cur[abnormal] = mean_val

            self.eff_global_pre = eff_global_cur.clone()

            # Compute prior (source: Global.cal_prior, line 133-135)
            prior = eff_global_cur.clone().detach()
            prior = (prior / prior.sum()).detach()

            # ===== Phase 2: Local training with ASD + DLA =====
            for client in self.selected_clients:
                client.set_parameters(self.global_model)
                client.train_with_prior(prior, r)

            # FedAvg aggregation (source: line 431)
            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFedYoYo)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
