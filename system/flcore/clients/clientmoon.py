import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client


class clientMOON(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.tau = args.tau
        self.mu = args.mu

        self.global_model = None
        self.old_model = copy.deepcopy(self.model)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
<<<<<<< HEAD

                # 获取表征和输出
                out = self.model(x)
                if isinstance(out, tuple):
                    rep, output = out[0], out[1]
                else:
                    rep = self.model.base(x)
                    if isinstance(rep, tuple):
                        rep = rep[0]
                    output = self.model.head(rep)

                loss = self.loss(output, y)

                # 获取历史模型和全局模型的表征
                with torch.no_grad():
                    out_old = self.old_model(x)
                    rep_old = out_old[0] if isinstance(out_old, tuple) else self.old_model.base(x)
                    if isinstance(rep_old, tuple):
                        rep_old = rep_old[0]
                    out_global = self.global_model(x)
                    rep_global = out_global[0] if isinstance(out_global, tuple) else self.global_model.base(x)
                    if isinstance(rep_global, tuple):
                        rep_global = rep_global[0]

=======
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                rep_old = self.old_model.base(x).detach()
                rep_global = self.global_model.base(x).detach()
>>>>>>> 15b6b60dba275c21157ead9a494232b7bb315b8d
                loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) / (torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) + torch.exp(F.cosine_similarity(rep, rep_old) / self.tau)))
                loss += self.mu * torch.mean(loss_con)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()
        self.old_model = copy.deepcopy(self.model)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_model = model

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
<<<<<<< HEAD

                # 获取表征和输出
                out = self.model(x)
                if isinstance(out, tuple):
                    rep, output = out[0], out[1]
                else:
                    rep = self.model.base(x)
                    if isinstance(rep, tuple):
                        rep = rep[0]
                    output = self.model.head(rep)

                loss = self.loss(output, y)

                # 获取历史模型和全局模型的表征
                out_old = self.old_model(x)
                rep_old = out_old[0] if isinstance(out_old, tuple) else self.old_model.base(x)
                if isinstance(rep_old, tuple):
                    rep_old = rep_old[0]
                out_global = self.global_model(x)
                rep_global = out_global[0] if isinstance(out_global, tuple) else self.global_model.base(x)
                if isinstance(rep_global, tuple):
                    rep_global = rep_global[0]

=======
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                rep_old = self.old_model.base(x).detach()
                rep_global = self.global_model.base(x).detach()
>>>>>>> 15b6b60dba275c21157ead9a494232b7bb315b8d
                loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) / (torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) + torch.exp(F.cosine_similarity(rep, rep_old) / self.tau)))
                loss += self.mu * torch.mean(loss_con)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num