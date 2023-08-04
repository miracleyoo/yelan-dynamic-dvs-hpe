import inspect
import importlib
import pytorch_lightning as pl
from dotdict import dotdict
from torch.utils.data import DataLoader
from ..utils import process_meta_files

class DataInterface(pl.LightningDataModule):
    def __init__(self, num_workers=8,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = dotdict(kwargs)
        self.gpus = self.kwargs.gpus
        self.load_data_module()

        processed = process_meta_files(
            self.kwargs.data_dir, 
            block_size=self.kwargs.seq_len, 
            base_number=self.kwargs.base_number, 
            test_characters=self.kwargs.test_characters,
            shuffle=True,
            random_offset=False,
            seed=self.kwargs.seed,
            remove_back_view=self.kwargs.remove_back_view,
            labels_processed=self.kwargs.labels_processed,
            percentile=self.kwargs.percentile,
            redundant_labels=self.kwargs.redundant_labels,
            front_only=self.kwargs.front_only,
            remove_db=self.kwargs.remove_db,
            remove_sb=self.kwargs.remove_sb,
            use_split_file=self.kwargs.use_split_file,
            rand_test=self.kwargs.rand_test,
            time_count_baseline=self.kwargs.time_count_baseline,
            occlusion=self.kwargs.occlusion,
            occlusion_rate=self.kwargs.occlusion_rate,
            occlusion_test=self.kwargs.occlusion_test,
            all_test=self.kwargs.all_test)
        
        if self.kwargs.all_test:
            test_indexes, test_readers = processed
            self.testset = self.instancialize(mode='test', shuffle=False,
                indexes=test_indexes, readers=test_readers)
        elif self.kwargs.test_characters is not None or self.kwargs.use_split_file:
            tv_indexes, test_indexes, tv_readers, test_readers = processed
            self.testset = self.instancialize(mode='test', shuffle=False,
                indexes=test_indexes, readers=test_readers)
        else:
            tv_indexes, tv_readers = processed

        if not self.kwargs.all_test:
            tv_length = len(tv_indexes)
            split_idx = int(tv_length*0.8) + (self.kwargs.seq_len - int(tv_length*0.8)%self.kwargs.seq_len)
            train_indexes = tv_indexes[:split_idx]
            # print(train_indexes.shape, train_indexes[:17],'\n\n\n\n\n')
            val_indexes = tv_indexes[split_idx:]
            # print(val_indexes[:20])
            self.trainset = self.instancialize(mode='train', shuffle=True, 
                indexes=train_indexes, readers=tv_readers)
            self.valset = self.instancialize(mode='val', shuffle=False,
                indexes=val_indexes, readers=tv_readers)
        
        print("Dataset Initialized.")
        
    def setup(self, stage=None):
        # Things to do on every accelerator in distributed mode
        # Assign train/val datasets for use in dataloaders
        pass

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.kwargs.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.kwargs.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=False)

    def test_dataloader(self):
        if self.kwargs.real_test and (self.kwargs.test_characters is not None or self.kwargs.use_split_file):
            return DataLoader(self.testset, batch_size=1, num_workers=self.num_workers, shuffle=False, pin_memory=False)
        else:
            return DataLoader(self.valset, batch_size=1, num_workers=self.num_workers, shuffle=False, pin_memory=False)

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        print(camel_name, )
        try:
            self.data_module = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, mode, shuffle, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.kwargs dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(mode, shuffle, **args1)
