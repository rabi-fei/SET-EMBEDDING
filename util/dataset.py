import torch
import random
import numpy as np
from torch.utils.data import Dataset


def generate_sets(dimension):
    ## dimension=3: [1] -> [1,0,0] -> "001" -> id=1(binary)
    idx_str = dict()
    idx_positive = dict()
    idx_negative = dict()
    str_idx = dict()
    for i in range(2**dimension):
        pos, neg = [], []
        id_str = bin(i)[2:].zfill(dimension)
        idx_str[i] =  id_str
        for index in range(len(id_str)):
            if id_str[index] == "1":
                pos.append(index)
            else:
                neg.append(index)
        idx_positive[i] = pos
        idx_negative[i] = neg
        str_idx[bin(i)[2:].zfill(dimension)] = i

    return idx_str, str_idx, idx_positive, idx_negative



class SetDataset(Dataset):
    "0-2**N"
    def __init__(self, dimension, device, sets=None):
        self.dimension = dimension
        self.device = device
        self.all_id = torch.arange(2**dimension, device=self.device)
        self.idx_str, self.str_idx, self.idx_pos, self.idx_neg = generate_sets(dimension)
        if not sets:
            self.sets = list(self.idx_str.values())
        else:
            self.sets = sets
        
    def __len__(self):
        # returns the length of the dataset
        return len(self.sets)

    def __getitem__(self, idx, pos_samples=1, neg_samples=1):
        # can't choose empty set including first and last.
        out_pos = torch.tensor([random.sample(self.idx_pos[id.item()], pos_samples) for id in idx], dtype=torch.long, device=self.device)
        out_neg = torch.tensor([random.sample(self.idx_neg[id.item()], neg_samples) for id in idx], dtype=torch.long, device=self.device)

        return out_pos, out_neg


class SetDatasetTest(Dataset):
    def __init__(self, dimension,train_dataset, device, sets=None):
        self.dimension = dimension
        self.device = device
        self.all_id = torch.arange(1, 2**dimension-1, device=self.device)
        self.base_dataset = train_dataset
        self.idx_str, self.str_idx, self.idx_pos, self.idx_neg = generate_sets(dimension)
        
    def __len__(self):
        # returns the length of the dataset
        return 2**self.dimension-2

    def __getitem__(self, id):
        # can't choose empty set including first and last.
        out_pos = torch.tensor(self.base_dataset.idx_pos[id], dtype=torch.long, device=self.device)
        out_neg = torch.tensor(self.base_dataset.idx_neg[id], dtype=torch.long, device=self.device)

        return out_pos, out_neg



class SetDatasetTensor(Dataset):
    #base: id
    #memory; id -> set-str
    #        id -> array

    def __init__(self, dimension, device="cpu"):
        self.dimension = dimension
        self.device = device
        self.idx_str = dict()
        self.all_id = torch.arange(2**dimension, device=self.device)
        set_array = []
        for idx in range(2 ** self.dimension):
            
            set_str = bin(idx)[2:].zfill(self.dimension)
            self.idx_str[idx] = set_str
            set_array.append([int(i) for i in set_str])
        set_array = np.array(set_array, dtype="float32")
        self.set_tensor = torch.from_numpy(set_array).to(device)

        print(self.set_tensor.dtype, self.set_tensor.device)

    def __len__(self):

        return 2**self.dimension

    def __getitem__(self, idx):
        out = torch.index_select(self.set_tensor, 0, idx)
        return out

class SetDatasetTrain_k(Dataset):
    #base: generate all sets which is less than k and denote is as [0, 1, 0, \cdots, 0]
    #memory; 01 repreent(tensor)

    def __init__(self, N, filted_size, device="cpu"):
        self.N = N
        self.filted_size = filted_size
        self.device = device
        self.idx_str = dict()
        set_array = []

        base_set = np.zeros((self.N, self.N))
        for i in range(self.N):
            base_set[i][i] = 1
        k_set_emb = base_set[:,:]
        sets_need = [k_set_emb]
        for i in range(self.filted_size):
#            print(i, k_set_emb.shape)
            generate = (base_set[None, :, :] + k_set_emb[:, None,:]).reshape(-1, self.N)
            delete_repeat = np.unique(generate, axis=0)
            k_set_emb = np.delete(delete_repeat, np.where(np.any(delete_repeat>1, axis=1)), axis=0)
        #    k_set_emb = torch.from_numpy(k_set_emb)
            sets_need.append(k_set_emb)
        
        self.sets = torch.from_numpy(np.concatenate(sets_need, axis=0)).to(device)



    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        out = torch.index_select(self.sets, 0, idx)
        return out


class SetDatasetTest_k(Dataset):
    #base: generate all sets which is less than k and denote is as [0, 1, 0, \cdots, 0]
    #memory; 01 repreent(tensor)

    def __init__(self, trainset):
        self.N = trainset.N
        self.filted_size = trainset.filted_size
        self.device = trainset.device

        self.sets_pos = dict()
        self.sets_neg = dict()
        for i in range(len(trainset.sets)):
                self.sets_pos[i] = torch.nonzero(trainset.sets[i]).squeeze(-1).tolist()
                self.sets_neg[i] = torch.nonzero(1-trainset.sets[i]).squeeze(-1).tolist()

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        pos_out = [self.sets_pos[i.item()] for i in idx]
        neg_out = [self.sets_neg[i.item()] for i in idx]
        return pos_out, neg_out
#print("fishied")
