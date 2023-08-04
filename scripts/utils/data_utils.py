import glob
import yaml
import json
import numpy as np
import os.path as op
from pathlib2 import Path

from .tore_utils import gen_tore_plus
from .file_utils import ToreSeqReader


def combine_meta_indexes(readers: dict, base_number: int = None):
    """ Combine multiple meta files' indexes into a single.
    Args:
        readers: dict, the readers.
        base_number: int, to make sure that the total data piece number is a
            multiple of base_number.
    Return:
        indexes: ndarray, shape: (?, 2), the first column means the reader's 
            index, and the second is the index inside this reader.
    """
    indexes = []
    print(f'===== READER NUM: {len(readers.metas)} =======')
    for i, meta in enumerate(readers.metas):
        total_num = meta['total_tore_count']
        if base_number is not None:
            total_num = total_num - total_num % base_number
        indexes.append(np.stack((i*np.ones(total_num, dtype=int), np.arange(total_num, dtype=int)), axis=-1))
    indexes = np.concatenate(indexes, axis=0)
    return indexes


def get_pair_by_idx(idx: int, indexes: np.ndarray, readers: dict, percentile: float = 80):
    """ Get the ntore and its corresponding label.
    Args:
        idx: int, the overall index.
        indexes: ndarray, shape: (?, 2), the merged indexes pack from all meta files.
        readers: dict, readers.
        percentile: float, the percentile used to generate the extra band in ntore.
    Return:
        ntore: the ntore processed.
        label: corresponding label.
    """
    reader_idx, tore_idx = indexes[idx]
    # TODO: When do the real run, change `get_pair_by_index_dummy` to `get_pair_by_index`
    tore, label = readers[reader_idx].get_pair_by_index(tore_idx)
    ntore = gen_tore_plus(tore, percentile=percentile)
    return ntore, label


def shuffle_arr_by_block(arr, block_size: int, ramdom_offset: bool = True, seed: int = None):
    """ Shuffle a list/ndarray in block.
    Args:
        arr: the input array to be shuffled.
        block_size: the block size used when shuffling.
        random_offset: whether to use add a random offset(0~block_size).
    Return:
        arr: shuffled array.
    """
    if type(arr) != np.ndarray:
        flag = True
        arr = np.array(arr)
    else:
        flag = False

    block_num = len(arr)//block_size
    if ramdom_offset:
        block_num -= 1
        offset = np.random.randint(0, block_size)
    keyidxs = np.arange(block_num)
    if seed is not None:
        print(f'[√] Using seed {seed} in train-val shuffling.')
        np.random.seed(seed)
    np.random.shuffle(keyidxs)
    new_idxs = np.repeat(keyidxs, block_size) * block_size
    addons = np.tile(np.arange(block_size), block_num)
    new_idxs += addons
    if ramdom_offset:
        new_idxs += offset
    res = arr[new_idxs]
    if flag:
        res = res.tolist()
    return res

def read_metas(meta_file_paths, labels_processed=False):
    metas = []
    for meta_file_path in meta_file_paths:
        with open(meta_file_path, 'r') as f:
            meta = json.load(f)
        meta['meta_file_dir'] = op.dirname(meta_file_path)
        if labels_processed:
            meta['label_file_dir'] = 'labels_processed'
        metas.append(meta)
    return metas

def process_meta_files(data_dir: str, block_size: int, base_number: int,
                       test_characters: list = None, shuffle: bool = True,
                       random_offset: bool = True, 
                       seed: int = 2333, remove_back_view: bool = False,
                       labels_processed:bool = False, percentile:int=90,
                       redundant_labels=False, front_only:bool=False,
                       remove_db=False, remove_sb=False,
                       rand_test=False, time_count_baseline=False, 
                       occlusion=False, occlusion_rate=1, occlusion_test=False,
                       all_test=False, use_split_file=True):
    """ Process all the meta files into a single indexes array, 
        and return the merged indexes and reader dicts.
    Args:
        data_dir: str,
        block_size: int, 
        base_number: int, 
        test_characters: list, the name list of characters who are used in test session.
        shuffle: bool, Whether to shuffle the train and validation indexes.
        random_offset: bool, whether to use add a random offset(0~block_size).
        seed: the random seed, used to guarantee the shuffled train val dataset split consistency. 
        remove_back_view: Don't use dataset that contains back view.
        labels_processed: Use the pre-processed label files.
        occlusion: Use the occluded dataset.
    """
    if use_split_file:
        if 'synthetic' in data_dir:
            filename = 'synthetic_randt.yaml' if rand_test else 'synthetic.yaml'
        elif 'ddhp' in data_dir:
            filename = 'ddhp_split.yaml'
        else:
            raise NotImplementedError
        print(f'[Info] Using split file {filename} to split the dataset.')
        with open(str(Path(data_dir).parent/filename), 'r') as f:
            data_pack = yaml.load(f, Loader=yaml.Loader)

        tv_meta_files = [op.join(data_dir, i, 'meta.json') for i in data_pack['names_tv']]
        test_meta_files = [op.join(data_dir, i, 'meta.json') for i in data_pack['names_test']]
        # print('[√] Cycling camera views. Using 1/4 of the dataset.')
        print(f"[Info] In total {len(tv_meta_files)} TV meta files, {len(test_meta_files)} test meta files.")

    else:
        meta_files = glob.glob(op.join(data_dir, '**', 'meta.json'))

        print(f"In total {len(meta_files)} files.")
        if remove_back_view:
            meta_files = [f for f in meta_files if 'back' not in Path(f).parts[-2]]
            print("[√] Back View data removed!")
        if remove_db:
            meta_files = [f for f in meta_files if 'DB' not in Path(f).parts[-2]]
            print("[√] Dynamic Background DDHP22 data removed!")
        if remove_sb:
            meta_files = [f for f in meta_files if 'SB' not in Path(f).parts[-2]]
            print("[√] Static Background DDHP22 data removed!")
        if front_only:
            meta_files = [f for f in meta_files if 'front' in Path(f).parts[-2]]
            print("[√] Back, Left, Right View data removed!")
        if labels_processed:
            print("[√] Using preprocessed labels.")

        # Filter out the test meta files
        if all_test:
            test_meta_files = meta_files
            tv_meta_files = []
        elif test_characters is not None:
            print("Test subjects: ", test_characters)
            test_meta_files = [f for f in meta_files if any([c in f for c in test_characters])]
            # Train and Validation meta files
            tv_meta_files = sorted(list(set(meta_files) - set(test_meta_files)))
        else:
            tv_meta_files = meta_files
        print("Len of meta files: ", len(meta_files))

    if not all_test:
        tv_readers = ToreSeqReader(tv_meta_files, labels_processed=labels_processed, percentile=percentile, redundant_labels=redundant_labels, 
                                    time_count_baseline=time_count_baseline, occlusion=occlusion, occlusion_rate=occlusion_rate)
        tv_indexes = combine_meta_indexes(tv_readers, base_number=base_number)
        if shuffle:
            tv_indexes = shuffle_arr_by_block(tv_indexes, block_size=block_size,
                                            ramdom_offset=random_offset, seed=seed)
        print("Train and Val indexes shape: ", tv_indexes.shape)

    if test_characters is None and not all_test:
        return tv_indexes, tv_readers

    
    print("Len of test meta files: ", len(test_meta_files))
    occlusion_test = occlusion_test & occlusion
    if occlusion_test:
        np.random.seed(2333)
        print("Adding fixed random occlusion.")
    test_readers = ToreSeqReader(test_meta_files, labels_processed=labels_processed, percentile=percentile, redundant_labels=redundant_labels, 
                                    time_count_baseline=time_count_baseline, occlusion=occlusion_test, occlusion_rate=1)
    test_indexes = combine_meta_indexes(test_readers, base_number=base_number)
    print("Test indexes shape: ", test_indexes.shape)

    if all_test:
        return test_indexes, test_readers
    elif test_characters is not None:
        return tv_indexes, test_indexes, tv_readers, test_readers