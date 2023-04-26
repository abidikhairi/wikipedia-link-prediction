import os
import numpy as np
import pandas as pd
import torch as th
from torch_geometric.data import InMemoryDataset, Data


class Wikipedia(InMemoryDataset):
    def __init__(self, root: str = None, transform = None, pre_transform = None, pre_filter = None, log: bool = True):
        super().__init__(root, transform, pre_transform, pre_filter, log)
        
    @property
    def num_features(self) -> int:
        return 300

    @property
    def raw_file_names(self):
        return ['n_feats_word2vec.npy', 'train.csv']
    
    def processed_file_names(self):
        return ['data_0.pt', 'data_1.pt', 'data_2.pt']
    
    def download(self):
        pass

    def process(self):
        feats_file = os.path.join(self.root, 'n_feats_word2vec.npy') 
        x = th.from_numpy(np.load(feats_file))
            
        edges_df = pd.read_csv(os.path.join(self.root, 'train.csv'))
        
        # start indexing from 0
        edges_df['id1'] = edges_df['id1'] - 1
        edges_df['id2'] = edges_df['id2'] - 1
        
        train_val_df = edges_df.sample(frac=0.9)
        test_df = edges_df.drop(train_val_df.index)
        train_df = train_val_df.sample(frac=0.7)
        valid_df = train_val_df.drop(train_df.index)

        for idx, df in enumerate((train_df, valid_df, test_df)):
            u = th.from_numpy(df['id1'].values).long()
            v = th.from_numpy(df['id2'].values).long()

            edge_label = th.from_numpy(df['label'].values).long()
            edge_index = th.stack((u, v))
        
            data = Data(x=x, edge_index=edge_index, edge_label=edge_label)

            th.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return 3

    def get(self, idx):
        data = th.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        
        return data


class WikipediaTest(InMemoryDataset):
    def __init__(self, root: str = None, transform = None, pre_transform = None, pre_filter = None, log: bool = True):
        super().__init__(root, transform, pre_transform, pre_filter, log)
        
    @property
    def num_features(self) -> int:
        return 300

    @property
    def raw_file_names(self):
        return ['n_feats_word2vec.npy', 'test.csv']
    
    def processed_file_names(self):
        return ['data_test.pt']
    
    def download(self):
        pass

    def process(self):
        feats_file = os.path.join(self.root, 'n_feats_word2vec.npy') 
        x = th.from_numpy(np.load(feats_file))
            
        edges_df = pd.read_csv(os.path.join(self.root, 'test.csv'))

        # start indexing from 0
        edges_df['id1'] = edges_df['id1'] - 1
        edges_df['id2'] = edges_df['id2'] - 1

        edge_id = th.from_numpy(edges_df['id'].values).long()
        u = th.from_numpy(edges_df['id1'].values).long()
        v = th.from_numpy(edges_df['id2'].values).long()

        edge_index = th.stack((u, v))
        
        data = Data(x=x, edge_index=edge_index, edge_id=edge_id)

        th.save(data, os.path.join(self.processed_dir, 'data_test.pt'))

    def len(self):
        return 1

    def get(self, *args, **kwargs):
        data = th.load(os.path.join(self.processed_dir, 'data_test.pt'))
        
        return data
