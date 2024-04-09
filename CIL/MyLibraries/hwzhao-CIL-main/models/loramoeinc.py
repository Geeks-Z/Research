import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.inc_net import LoRAMoEIncNet, MultiBranchCosineIncrementalNet

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = LoRAMoEIncNet(args, True)
        self.batch_size = args["batch_size"] if args["batch_size"] is not None else 128
        self.init_lr = args["init_lr"] if args["init_lr"] is not None else 0.01
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

    def after_task(self):
        self._known_classes = self._total_classes
        # 固定上一阶段的lora_expert参数更新
        self.frozen_lora_expert()
        self._network.backbone.cur_task = self._cur_task + 1;

    def replace_fc(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto
        return model

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                              source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        self._network.to(self._device)
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,
                                  weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'],
                                                         eta_min=self.min_lr)
        self._init_train(train_loader, test_loader, optimizer, scheduler,self._cur_task)
        self.replace_fc(train_loader_for_protonet, self._network, None)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler,cur_task):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out = self._network(inputs)
                logits = out["logits"]
                gate_loss = out["gate_loss"]
                # self._network.backbone.gate_loss
                loss = F.cross_entropy(logits, targets)+(self._network.backbone.blocks[0].adaptmlp.gate_loss
                        +self._network.backbone.blocks[1].adaptmlp.gate_loss
                        +self._network.backbone.blocks[2].adaptmlp.gate_loss
                        +self._network.backbone.blocks[3].adaptmlp.gate_loss
                        +self._network.backbone.blocks[4].adaptmlp.gate_loss
                        +self._network.backbone.blocks[5].adaptmlp.gate_loss
                        +self._network.backbone.blocks[6].adaptmlp.gate_loss
                        +self._network.backbone.blocks[7].adaptmlp.gate_loss
                        +self._network.backbone.blocks[8].adaptmlp.gate_loss
                        +self._network.backbone.blocks[9].adaptmlp.gate_loss
                        +self._network.backbone.blocks[10].adaptmlp.gate_loss
                        +self._network.backbone.blocks[11].adaptmlp.gate_loss)/12
                # loss = gate_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def frozen_lora_expert(self):
        frozen_lora_down = str(self._network.backbone.cur_task)+'.down'
        frozen_lora_up = str(self._network.backbone.cur_task)+'.up'
        for name, p in self._network.named_parameters():
            if frozen_lora_down in name or frozen_lora_up in name:
                p.requires_grad = False


