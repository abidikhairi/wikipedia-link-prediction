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
        return ['data.pt']
    
    def download(self):
        pass

    def process(self):
        feats_file = os.path.join(self.root, 'n_feats_word2vec.npy') 
        x = th.from_numpy(np.load(feats_file))
            
        edges_df = pd.read_csv(os.path.join(self.root, 'train.csv'))

        # start indexing from 0
        edges_df['id1'] = edges_df['id1'] - 1
        edges_df['id2'] = edges_df['id2'] - 1

        u = th.from_numpy(edges_df['id1'].values).long()
        v = th.from_numpy(edges_df['id2'].values).long()

        edge_label = th.from_numpy(edges_df['label'].values).long()
        edge_index = th.stack((u, v))
        
        data = Data(x=x, edge_index=edge_index, edge_label=edge_label)

        th.save(data, os.path.join(self.processed_dir, 'data.pt'))

    def len(self):
        return 1

    def get(self, *args, **kwargs):
        data = th.load(os.path.join(self.processed_dir, 'data.pt'))
        
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
