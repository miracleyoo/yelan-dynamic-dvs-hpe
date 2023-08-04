import os.path as op
import numpy as np

class MaskSeqReader:
    NAME_DICT = {
        0.02: '0.02.npz',
        5: '5.npz'
    }
    def __init__(self, root: str, acc_time: float = 0.02) -> None:
        self.data_path = op.join(root, self.NAME_DICT[acc_time])
        self.data = None

    @property
    def total_frame_count(self):
        return len(self.data.keys())

    def load_data(self):
        self.data = np.load(self.data_path, allow_pickle=True)

    def read_acc_frame(self, acc_frame_index: int):
        if self.data is None:
            self.load_data()
        frame = self.data.get(str(acc_frame_index))
        return frame

    def clear_cache(self):
        self.data = None