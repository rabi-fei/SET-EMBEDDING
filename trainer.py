import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_dataset, test_dataset=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.max_epochs = 750
        self.batch_size = 256

    def save_checkpoint(self):
        pass

    def train(self):
        model = self.model
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.005, betas=(0.9, 0.95))

        def test():
            sets_pos = self.test_dataset.sets_pos
            sets_neg = self.test_dataset.sets_neg
            length = len(sets_pos)
            test_batch_size = 256
            test_index = torch.arange(length, device=self.test_dataset.device)

            step = 0
            mrrs = []
            while step < length:
                test_data = test_index[step:min(length, step+test_batch_size)]
                set_emb = self.model.set_emb(test_data)
                all_ent_emb = F.relu(self.model.ent_emb.weight)
                distance = torch.matmul(set_emb, all_ent_emb.transpose(-2, -1))
                pos_index, neg_index = self.test_dataset[test_data]
                argsort = torch.argsort(distance, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                ranking = ranking.scatter_(
                        1, argsort, torch.arange(self.model.N, device=self.test_dataset.device).to(torch.float).repeat(argsort.shape[0], 1))
                for sub_i in range(len(test_data)):
                    i = step + sub_i
                    cur_ranking, indices = torch.sort(ranking[sub_i, sets_pos[i]])
                    ranking_pos = cur_ranking - torch.arange(len(sets_pos[i]),  device=self.test_dataset.device) + 1
                    mrr = torch.mean(1. / ranking_pos)
                    mrrs.append(mrr)
                step += test_batch_size
            avg_mrr = torch.stack(mrrs).mean()

                
            
            return avg_mrr

        def run_epoch(mode):
            length = len(self.train_dataset)
            data_index = torch.arange(length,device=self.train_dataset.device)
            shuffle_index = torch.randperm(len(data_index))
            # generate positive samples and negative samples
            all_train_data = data_index[shuffle_index]
            step = 0
            collect_loss = []
            while step < len(all_train_data):
                train_data = all_train_data[step : step+self.batch_size]
                set_emb = self.model.set_emb(train_data)
                ent_emb = F.relu(self.model.ent_emb.weight)
                target = self.train_dataset[train_data]
                loss = self.model(set_emb, ent_emb, target)
               # Loss for positive and negative 
                loss.backward()
                collect_loss.append(loss.item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                model.zero_grad()

                step += self.batch_size
            print("-"*120)
            print(f"this epoch's loss:{sum(collect_loss)/len(collect_loss)}")


        for epoch in range(self.max_epochs):

            run_epoch('train')
            if self.test_dataset is not None and epoch % 25 ==0 :
                avg_mrr = test()
                print(f"test_score:{avg_mrr}")
        all_ent_emb = F.relu(self.model.ent_emb.weight.data)
        union_set = self.model.get_dis_emb([self.model.set_emb.weight[0], self.model.set_emb.weight[1]])
        decode = torch.sigmoid(torch.matmul(self.model.set_emb.weight[:2], all_ent_emb.transpose(-2, -1)))
        return avg_mrr
        self.save_checkpoint()

