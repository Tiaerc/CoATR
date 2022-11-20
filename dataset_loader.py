import torch
import numpy as np
import os
import random
from torch.utils.data import Dataset
from scipy.io import loadmat
import scipy.io as sio
np.random.seed(1000)



def load_dataset(mat_path, MR):
    data = torch.from_numpy(sio.loadmat(mat_path)['out'] / 1.0)
    dim1, dim2, dim3 = data.shape
    node_day_is_miss = np.round(np.random.rand(dim1, dim2) + 0.5 - MR)
    miss_pos = np.argwhere(node_day_is_miss == 0.0) #可以除去缺失值本来就是0的
    miss_pos_list = miss_pos.tolist()
    miss_pos_set = set()
    for (i, j) in miss_pos_list:
        miss_pos_set.add((i, j))

    train_idx = []
    test_idx = []
    train_label = []
    test_label = []
    train_index_true = []
    test_index_true = []

    for i in range(dim1):
        for j in range(dim2):
            idxs = []
            datas = []
            idxs_true = []
            for k in range(dim3):
                idxs_true.append(dim2*dim3 * i + dim3 * j + k)
                idx = [i, j, k]
                idxs.append(idx)
                datas.append(data[i, j, k])

            if (i, j) in miss_pos_set: # test_test
                test_idx.append(idxs)
                test_index_true.append(idxs_true)
                test_label.append(datas)
            else: # train_set
                train_idx.append(idxs)
                train_label.append(datas)
                train_index_true.append(idxs_true)

    train_idx = np.array(train_idx)
    train_index_true = np.array(train_index_true)
    test_idx = np.array(test_idx)

    train_label = np.array(train_label)
    test_index_true = np.array(test_index_true)
    test_label = np.array(test_label)

    train_dataset = HSIDataset(train_idx, train_label, train_index_true)
    test_dataset = HSIDataset(test_idx, test_label, test_index_true)

    return train_dataset, test_dataset, data


class HSIDataset(Dataset):
    def __init__(self, idx, label, idx_true):
        super().__init__()
        self.idx = idx
        self.label = label
        self.idx_true = idx_true

    def __getitem__(self, index):
        return self.idx[index], self.label[index]

    def __len__(self):
        return len(self.label)

if __name__ == "__main__":
    train_dataset, test_dataset = load_dataset('Traffic/GZ.mat', 0.3)
    pass
