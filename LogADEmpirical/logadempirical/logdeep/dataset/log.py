#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class log_dataset(Dataset):
    def __init__(self, logs, labels, window_size):
        # self.logs = []
        # for i in range(len(labels)):
        #     features = [torch.tensor(logs[i][0][0], dtype=torch.long)]
        #     for j in range(1, len(logs[i][0])):
        #         features.append(torch.tensor(logs[i][0][j], dtype=torch.float))
        #     self.logs.append({
        #         "features": features,
        #         "idx": logs[i][1]
        #     })
        # self.labels = labels

        self.logs = []
        for i in range(len(labels)):
            log_re = np.reshape(logs[i][1:-1], (window_size, 300))
            features = [torch.tensor([], dtype=torch.long), torch.tensor([logs[i][0]], dtype=torch.long)]
            feature_list = []
            for j in range(0, len(log_re)):
                feature_list.append(log_re[j])
            features.append(torch.tensor(feature_list, dtype=torch.float))
            self.logs.append({
                "features": features,
                "idx": int(logs[i][-1])
            })
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.logs[idx], self.labels[idx]


if __name__ == '__main__':
    data_dir = '../../data/'
