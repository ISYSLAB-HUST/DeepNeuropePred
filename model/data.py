
import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def find_file(feature_dir):
    file_name_list = []
    for item in glob(os.path.join(feature_dir, '*.npz')):
        file_name_list.append(item)
    return file_name_list

class NeuroPepDataset(Dataset):
    """custom dataset
    """

    def __init__(self, feature_dir):
        self.file_name_list = find_file(feature_dir)

    def __getitem__(self, index):
        _temp = np.load(self.file_name_list[index])
        return _temp['feature'], _temp['pos']

    def __len__(self):
        return len(self.file_name_list)



def collate_fn(batch_data):
    """pretrain padding func

    Args:
        batch_data (list): tensor
    """
    batch_data.sort(key=lambda x: x[0].shape[-1], reverse=True)
    feature_matrix = []
    file_site = []
    for row in batch_data:
        feature_matrix.append(torch.from_numpy(row[0]))
        file_site.append(row[-1].item())
    feature_matrix = pad_sequence(feature_matrix, batch_first=True, padding_value=0)

    return feature_matrix, file_site