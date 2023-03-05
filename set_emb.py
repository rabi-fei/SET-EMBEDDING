import torch
import torch.nn as nn
import torch.nn.functional as F

class SetEmbedding(nn.Module):

    def __init__(self, N, sets_number, emb_d):
        super().__init__()
        self.set_number = sets_number
        self.N = N
        self.set_emb = nn.Embedding(self.set_number, emb_d)
        self.ent_emb = nn.Embedding(N, emb_d)
        self.set_emb.weight.data.normal_(mean=0.0, std=0.02)
        self.ent_emb.weight.data.normal_(mean=1.0, std=0.02)

    def get_ent_emb(self, idx):

        pos_emb = F.relu(self.ent_emb(idx))

        return pos_emb

    def get_dis_emb(self, embs):
        constant = torch.log(torch.tensor(2.0, device=embs[0].device))
        before_dis = torch.stack(embs, dim=0)
        after_dis = torch.logsumexp(before_dis, dim=0) -  constant
        return after_dis
    
    def forward(self, set_emb, ent_emb, target):
        MCEloss = nn.MultiLabelSoftMarginLoss(reduction='sum')
        nllloss_func=nn.NLLLoss()
#        dots = torch.max((set_emb.unsqueeze(1) * ent_emb.unsqueeze(0)), dim=-1)[0]
        dots = torch.matmul(set_emb, ent_emb.transpose(-2, -1))
        logits = -F.logsigmoid(dots)
        new_target = (target - 1/2) * 2
        loss = -(target*F.logsigmoid(dots) + (1-target)*F.logsigmoid(-dots)).sum()
#        loss = MCEloss(dots, target)
        

        return loss

