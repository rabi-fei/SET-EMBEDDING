import numpy as np
import pandas as pd
from util.dataset import SetDataset, SetDatasetTest, SetDatasetTrain_k, SetDatasetTest_k
from set_emb import SetEmbedding
from trainer import Trainer


def Max_mrr_socre(dimension):
    mrr_score_max = dict(dict(zip(range(1, dimension), [0]*(dimension-1))))
    for i in range(1, dimension):
        score_k = sum([1 / i for i in range(1, i+1)]) / i
        mrr_score_max[i] = score_k
    return mrr_score_max
    
# ToDo: 1.batch MRR
#       2.new opt


N_ = 10
#filted_size = 4
N_s = [4, 8, 12, 16, 20, 24]
k_s = [4, 8, 12, 16, 20, 24]
outfloder = "resluts/"
outfile = outfloder + "scores.csv"
emb_dim_s = [4, 8, 12, 16, 20, 24]
scores = np.zeros((len(N_s), len(k_s), len(emb_dim_s)))
for index_N in range(len(N_s)):
    for index_k in range(len(k_s)):
        for index_d in range(len(emb_dim_s)):
    #        max_mrr_socre = Max_mrr_socre(N)
            if index_d > index_N:
                continue
            N, k, emb_dim = N_s[index_N], k_s[index_k], emb_dim_s[index_d]
            train_dataset = SetDatasetTrain_k(N, k, "cuda")
            test_dataset = SetDatasetTest_k(train_dataset)
            model = SetEmbedding(N, len(train_dataset), emb_dim)
            model.cuda()

            trainer = Trainer(model, train_dataset, test_dataset)
            last_mrr = trainer.train()
            scores[index_N][index_k][index_d] = last_mrr
            pd.DataFrame(scores.reshape(-1, len(emb_dim_s))).to_csv(outfile)


print("train fished")