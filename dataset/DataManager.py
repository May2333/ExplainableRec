import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
# from prefetch_generator import BackgroundGenerator

# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())

class RecDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # self.users, self.items, self.ratings, self.tips = zip(*(sorted(zip(self.users, self.items, self.ratings, self.tips), key=lambda x:x[1])))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset[index][1], self.dataset[index][2], self.dataset[index][3]


class RecDatasetManager(object):
    def __init__(self, opt):
        self.dataset_path = opt["dataset_path"]
        self.batch_size = opt["batch_size"]
        self.toolkits = opt["toolkits"]
        self.device = opt["device"]
        self.max_tip_len = opt["max_tip_len"]
        with open(self.dataset_path, 'rb') as fd:
            train_raw, val_raw, test_raw = pickle.load(fd)
        self.train_dataset = RecDataset(train_raw)
        self.val_dataset = RecDataset(val_raw)
        self.test_dataset = RecDataset(test_raw)
        self.train_loader = DataLoader(dataset=self.train_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(dataset=self.val_dataset, shuffle=False, batch_size=self.batch_size, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(dataset=self.test_dataset, shuffle=False, batch_size=self.batch_size, collate_fn=self.collate_fn_test)

    def collate_fn(self, batch):
        '''
            change tips to padding vectors
        '''
        users = []
        items = []
        ratings = []
        tips_vecs = []
        for u, i, s, t in batch:
            users.append(u)
            items.append(i)
            ratings.append(s)
            tips_vecs.append(self.toolkits.text2vec(t.lower(), add_end=True))
        tips_lens = np.array([len(t) for t in tips_vecs])
        tip_pad_len = min(max(tips_lens), self.max_tip_len)
        tips_pad_vecs = self.toolkits.pad2d(tips_vecs, tip_pad_len)

        tips_sorted_idx = np.argsort(tips_lens)[::-1]
        users = np.array(users)
        items = np.array(items)
        ratings = np.array(ratings).astype(np.float32)
        tips_pad_vecs = tips_pad_vecs

        return  (users, items, torch.from_numpy(ratings).to(self.device), torch.from_numpy(tips_pad_vecs).to(self.device))
        # pass

    def collate_fn_test(self, batch):
        '''
            change tips to padding vectors
        '''
        users = []
        items = []
        ratings = []
        tips_vecs = []
        tips = []
        for u, i, s, t in batch:
            users.append(u)
            items.append(i)
            ratings.append(s)
            tips_vecs.append(self.toolkits.text2vec(t, add_end=True))
            tips.append(t.lower() + '\n')
        tips_lens = np.array([len(t) for t in tips_vecs])
        tip_pad_len = min(max(tips_lens), self.max_tip_len)
        tips_pad_vecs = self.toolkits.pad2d(tips_vecs, tip_pad_len)

        users = np.array(users)
        items = np.array(items)
        ratings = np.array(ratings).astype(np.float32)

        return  (users, items, torch.from_numpy(ratings).to(self.device), torch.from_numpy(tips_pad_vecs).to(self.device), tips)
        # pass


class NarreDatasetManager(object):
    def __init__(self, opt):
        self.dataset_path = opt["dataset_path"]
        self.batch_size = opt["batch_size"]
        self.toolkits = opt["toolkits"]
        self.device = opt["device"]
        self.max_tip_len = opt["max_tip_len"]
        self.user_nums = len(self.toolkits.user2ind_dict)
        self.item_nums = len(self.toolkits.item2ind_dict)
        with open(opt["relations_path"], "rb") as f_relations:
            self.user_ne_items, self.item_ne_users, _, _, \
                _, _, _ = pickle.load(f_relations)
        with open(opt["graph_info_path"], "rb") as f_ginfo:
            _, _, self.tip2text_vectors, self.review2text_vectors = pickle.load(f_ginfo)
        with open(self.dataset_path, 'rb') as fd:
            train_raw, val_raw, test_raw = pickle.load(fd)
        self.train_dataset = RecDataset(train_raw)
        self.val_dataset = RecDataset(val_raw)
        self.test_dataset = RecDataset(test_raw)
        self.train_loader = DataLoader(dataset=self.train_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(dataset=self.val_dataset, shuffle=False, batch_size=self.batch_size, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(dataset=self.test_dataset, shuffle=False, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        '''
            change tips to padding vectors
        '''
        users = []
        items = []
        users_neighbors = []
        items_neighbors = []
        users_tips_vec = []
        items_tips_vec = []
        ratings = []
        tips_vecs = []
        for u, i, s, t in batch:
            users.append(u)
            items.append(i)
            users_neighbors.append(self.user_ne_items[u])
            items_neighbors.append(self.item_ne_users[i])
            users_tips_vec.append([self.tip2text_vectors[(u, ne_i)][0] + [self.toolkits.end_ind] for ne_i in self.user_ne_items[u]])
            items_tips_vec.append([self.tip2text_vectors[(ne_u, i)][0] + [self.toolkits.end_ind] for ne_u in self.item_ne_users[i]])
            ratings.append(s)
            tips_vecs.append(self.toolkits.text2vec(t.lower(), add_end=True))
        tips_lens = np.array([len(t) for t in tips_vecs])
        tip_pad_len = min(max(tips_lens), self.max_tip_len)
        tips_pad_vecs = self.toolkits.pad2d(tips_vecs, tip_pad_len)
        users_pad_neighbors = self.toolkits.pad2d(users_neighbors, 25, self.item_nums)
        items_pad_neighbors = self.toolkits.pad2d(items_neighbors, 25, self.user_nums)
        users_pad_tips_vec = self.toolkits.pad3d(users_tips_vec, 25, self.max_tip_len)
        items_pad_tips_vec = self.toolkits.pad3d(items_tips_vec, 25, self.max_tip_len)

        users = np.array(users)
        items = np.array(items)
        ratings = np.array(ratings).astype(np.float32)

        return  (users, items,
                torch.from_numpy(ratings).to(self.device),
                torch.from_numpy(tips_pad_vecs).to(self.device),
                torch.from_numpy(users_pad_tips_vec).to(self.device),
                torch.from_numpy(items_pad_tips_vec).to(self.device),
                torch.from_numpy(users_pad_neighbors).to(self.device),
                torch.from_numpy(items_pad_neighbors).to(self.device))
        # pass