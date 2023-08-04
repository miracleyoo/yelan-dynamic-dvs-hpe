import os.path as op
import numpy as np
import json
import gc
# from memory_profiler import profile
from ..skeleton_helpers import prepare_label
from  ..cv_helpers import get_random_mask
class ToreSeqReader:
    def __init__(self, meta_file_paths, 
                 labels_processed=False, 
                 percentile=90, 
                 redundant_labels=False, 
                 time_count_baseline=False,
                 occlusion=False,
                 occlusion_rate=1):
        """
        :param meta_file_path: path to the json file containing meta information
        """
        self.labels_processed = labels_processed
        self.meta_file_paths = meta_file_paths
        self.percentile = percentile
        self.redundant_labels = redundant_labels
        self.time_count_baseline = time_count_baseline
        self.occlusion = occlusion
        self.occlusion_rate = occlusion_rate
        self.necessary_keys = ['skeleton', 'normalized_skeleton', 'camera', 'M', 'z_ref', 'mask']
        self.metas = []
        for meta_file_path in meta_file_paths:
            with open(meta_file_path, 'r') as f:
                meta = json.load(f)
            meta['meta_name'] = meta_file_path.split(op.sep)[-2]
            meta['meta_file_dir'] = op.dirname(meta_file_path)
            if labels_processed:
                meta['label_file_dir'] = 'labels_processed'
            self.metas.append(meta)

        self.current_tore_batch = None
        self.current_label_batch = None
        self.current_batch_index = (-1, -1)

    def load_batch(self, meta_idx:int, idx: int):
        meta = self.metas[meta_idx]
        batch_idx = idx // meta['batch_size']
        self.meta_name = meta['meta_name']
        mid_dir_name = meta['tore_file_dir'] if not self.time_count_baseline else 'frames'
        
        if (meta_idx, batch_idx) != self.current_batch_index:
            self.current_tore_batch = np.load(op.join(meta['meta_file_dir'],
                                                mid_dir_name,
                                                meta['tore_file_list'][str(batch_idx)]), allow_pickle=True)
            self.current_label_batch = np.load(op.join(meta['meta_file_dir'],
                                                meta['label_file_dir'],
                                                meta['label_file_list'][str(batch_idx)]), allow_pickle=True)
            self.current_batch_index = (meta_idx, batch_idx)

        
    def get_tore_by_index(self, meta_idx:int, idx: int) -> np.ndarray:
        self.load_batch(meta_idx, idx)
        tore = self.current_tore_batch[str(idx)]
        if self.time_count_baseline:
            tore = (tore / 255).astype(np.float16)
        return tore

    def get_label_by_index(self, meta_idx:int, idx: int) -> dict:
        self.load_batch(meta_idx, idx)
        return self.current_label_batch[str(idx)].item()

    def get_pair_by_index(self, meta_idx:int, idx: int):
        return self.get_tore_by_index(meta_idx, idx), self.get_label_by_index(meta_idx, idx)

    def load_seq_by_index(self, meta_idx:int, idx: int, seq_len:int=16):
        tores = []
        poses = []
        labels = {}
        self.load_batch(meta_idx, idx)

        for i in range(seq_len):
            tore = self.current_tore_batch[str(idx+i)]
            label = self.current_label_batch[str(idx+i)].item()
            if not self.labels_processed:
                label = prepare_label(label)
            poses.append(label)
            tores.append(tore)
        tores = np.stack(tores)
        if self.occlusion and np.random.rand() < self.occlusion_rate:
            tores = get_random_mask(tores)
        if self.time_count_baseline:
            tores = (tores / 255).astype(np.float16)
        selected_keys = list(poses[0].keys()) if self.redundant_labels else self.necessary_keys
        for key in selected_keys:
            labels[key] = np.stack([pose[key] for pose in poses])
            if key != 'mask':
                labels[key] = labels[key].astype(np.float32)
        labels['name'] = self.meta_name

        self.cleanup()
        return tores, labels

    def cleanup(self):     
        self.current_label_batch.close()
        self.current_tore_batch.close()
        self.current_label_batch = None
        self.current_tore_batch = None
        self.current_batch_index = (-1, -1)
        gc.collect()
