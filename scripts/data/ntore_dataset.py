from torch.utils.data import Dataset

class NtoreDataset(Dataset):
    def __init__(self, mode, shuffle, batch_size, indexes, readers,
                 frame_size, n_joints, percentile, test_characters, seq_len, partial_dataset):

        self.readers = readers
        self.indexes = indexes
        self.seq_len = seq_len
        self.partial_dataset = partial_dataset if mode != 'test' else 1

    def __len__(self):
        return int(self.partial_dataset*(len(self.indexes)//self.seq_len))

    def __getitem__(self, idx):
        meta_idx, tore_idx = self.indexes[idx*self.seq_len]
        try:
            ntores, labels = self.readers.load_seq_by_index(meta_idx, tore_idx, seq_len=self.seq_len)
            return ntores, labels
        except Exception as e:
            print(e)
            print(meta_idx, tore_idx)
            print(self.readers.metas[meta_idx])
            print(self.readers.metas[meta_idx]['meta_name'])
            print(self.indexes[idx*self.seq_len])
            raise e
