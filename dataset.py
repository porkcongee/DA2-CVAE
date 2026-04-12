import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib

class ProteinDataset(Dataset):
    def __init__(self, data_file, task, N_clusters, S=10000):
        assert os.path.exists(data_file), f'{data_file} not exists'
        self.raw_data = np.load(data_file)
        self.atom_num = self.raw_data.shape[1] // 3
        print(f'ATOM NUM: {self.atom_num}')

        kmeans_name = f'{task}/kmeans_{task}_n{N_clusters}.npy'
        assert os.path.exists(kmeans_name), f'{kmeans_name} not exists'
        self.kmeans = np.load(kmeans_name)
        self.kmeans_oh = np.eye(N_clusters)[self.kmeans]
        self.c_nums = N_clusters

        assert N_clusters - 1 in self.kmeans, f'{N_clusters} not exists'
        
        for c in range(N_clusters):
            c_list = np.where(self.kmeans == c)
            c_num = c_list[0].size
            print(f'C{c} NUM:{c_num}')
            c_file = f'{task}/cluster_{task}_n{N_clusters}_{c}_{S}.npy'
            if os.path.exists(c_file):
                c_select = np.load(c_file)
            else:
                if c_num < S:
                    S_tmp = S % c_num
                    c_select = np.random.choice(c_num, size=S_tmp, replace=False)
                    for _ in range(S // c_num):
                        c_select = np.concatenate((np.arange(c_num),c_select))
                else:
                    c_select = np.random.choice(c_num, size=S, replace=False)
                np.save(c_file, c_select)
            c_data = self.raw_data[c_list[0],:][c_select,:]
            label = self.kmeans_oh[c_list[0],:][c_select,:]
            if c == 0:
                self.all_data = c_data
                self.label_data = label
            else:
                self.all_data = np.concatenate((self.all_data, c_data),axis=0)
                self.label_data = np.concatenate((self.label_data, label),axis=0)
            print(self.all_data.shape, self.label_data.shape)

        self.data_num = self.all_data.shape[0]
        self.minmax_scaler = MinMaxScaler(clip=True)
        self.all_data = self.minmax_scaler.fit_transform(self.all_data)
        
        self.data = np.concatenate((self.all_data, self.label_data), axis=1)
        print(self.data.shape)
        
        

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        data = self.data[idx]
        data = torch.from_numpy(data).float()
        return data

    def save_scaler(self, name):
        joblib.dump(self.minmax_scaler, f'{name}.pkl')
        return


if __name__ == '__main__':
    task = 'example'
    data_file = f'raw_data/coor_xyz_{task}.npy'
    dataset = ProteinDataset(data_file, task, 4, 10000)
