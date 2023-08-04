import os
import json
from typing import Any
import numpy as np


def read_required_field(meta_info: dict, field: str) -> Any:
    if meta_info[field]:
        return meta_info[field]
    else:
        raise Exception("Required field '{}' missing from meta file".format(field))


class ImgSeqReader:

    def __init__(self, meta_file_path: str, loop_read: bool = False, acc_time: float = 0.02,
                 cache_size: int = 1) -> None:
        """
        :param meta_file_path: path to the json file containing meta information
        :param loop_read: usually used for the background video, when using read_next() and the end of the frames is
                          reached, read backward and vice versa.
        :param acc_time: use to accumulate the alpha channel (mask), if a pixel is marked positive in any of the frame
                         within the accumulation period, it will be positive in the accumulation frame.
        :param cache_size: the size of the queue storing historical used batch file,
                           loading a batch from the queue is faster but takes memory
        """
        # frame_count and current_frame_index are the same when loop_read is False
        self.frame_count: int = 0
        self.current_frame_index: int = 0
        self.current_batch = None
        self.current_batch_index: int = -1
        self.loop_read: bool = loop_read
        self.file_idx: int = 0
        self.reverse_reading = False
        self.acc_time: float = acc_time
        self.meta_info: dict = {}
        self.total_frame_count: int = 0
        self.fps: int = 0
        self.batch_size: int = 0
        self.file_list: dict = {}

        self.acc_frame_index: int = 0

        self.unpack_meta(meta_file_path)
        self.meta_file_dir: str = os.path.dirname(meta_file_path)

        if self.acc_time < 1 / self.fps:
            print("Warning: Set accumulation time is smaller than the frame time precision. Ignore this message if "
                  "not using accumulation frame.")

        self.acc_frame_not_aligned: bool = not (acc_time * self.fps).is_integer()
        if self.acc_frame_not_aligned:
            # 啊算了不写了这块反正也用不上
            print(
                "Number of frame in the accumulation interval is not an integer. Accumulation frame might be "
                "calculated from different number of original frames")
            print("Not Supported Yet")

        if cache_size <= 0:
            raise ValueError("Cache size must be greater or equal to 1.")
        self.cache_size = cache_size
        self.cache_queue = []
        self.cache_dict = {}

    def unpack_meta(self, meta_file_path: str) -> None:

        if not meta_file_path:
            raise Exception("No file input provided")

        if not os.path.isfile(meta_file_path):
            raise FileNotFoundError("Input File: '{}' doesn't exist".format(meta_file_path))

        with open(meta_file_path, 'r') as input_file:
            self.meta_info = json.load(input_file)

        self.total_frame_count = read_required_field(self.meta_info, "frame_count")
        self.fps = read_required_field(self.meta_info, "fps")
        self.batch_size = read_required_field(self.meta_info, "batch_size")
        self.file_list = read_required_field(self.meta_info, "file_list")

    def total_frame(self) -> int:
        return self.total_frame_count

    def load_batch_from_disk(self, batch_index: int) :
        batch_file = np.load(os.path.join(self.meta_file_dir, self.file_list[str(batch_index)]), allow_pickle=True)
        self.current_batch = batch_file[list(batch_file.keys())[0]]
        self.current_batch_index = batch_index

    def load_batch_from_queue(self, batch_index: int) -> None:
        if batch_index in self.cache_queue:
            self.current_batch = self.cache_dict[batch_index]
            self.current_batch_index = batch_index
            self.cache_queue.remove(batch_index)
        else:
            if len(self.cache_queue) >= self.cache_size:
                removed_batch_index = self.cache_queue.pop(0)
                del self.cache_dict[removed_batch_index]
            self.load_batch_from_disk(batch_index)
            self.cache_dict[batch_index] = self.current_batch
        self.cache_queue.append(batch_index)

    def load_batch(self, batch_index: int) -> None:
        if self.cache_size <= 1:
            self.load_batch_from_disk(batch_index)
        else:
            self.load_batch_from_queue(batch_index)

    def read_next_frame(self) -> (np.ndarray, float):
        time_stamp: float = self.frame_count / self.fps
        frame_data = self.read_frame(self.current_frame_index)
        self.frame_count += 1

        if self.reverse_reading:
            self.current_frame_index -= 1
        else:
            self.current_frame_index += 1

        if self.loop_read and (self.current_frame_index >= self.total_frame_count - 1 or self.current_frame_index == 0):
            self.reverse_reading = not self.reverse_reading

        return frame_data, time_stamp

    def read_frame(self, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            raise IndexError("Frame index must be greater or equal to 0. Received: {}".format(frame_index))
        if frame_index >= self.total_frame_count:
            raise IndexError("Frame index out of bound. Image sequence has {} frames. Received: {}".format(self.total_frame_count, frame_index))

        if int(frame_index / self.batch_size) != self.current_batch_index:
            self.load_batch(int(frame_index / self.batch_size))

        return self.current_batch[frame_index % self.batch_size]

    def read_next_acc_frame(self):
        self.acc_frame_index += 1
        return self.read_acc_frame(self.acc_frame_index - 1)

    def read_acc_frame(self, acc_frame_index: int):
        start_time = acc_frame_index * self.acc_time
        end_time = start_time + self.acc_time

        start_frame_index = int(start_time * self.fps)
        end_frame_index = int(end_time * self.fps) - 1

        if end_frame_index >= self.total_frame_count:
            end_frame_index = self.total_frame_count - 1
        if self.acc_frame_not_aligned:
            return None
        else:
            start_frame_index += 1

        acc_frame = None
        for frame in range(start_frame_index, end_frame_index + 1):
            frame_data = self.read_frame(frame)
            if acc_frame is not None:
                acc_frame += frame_data
            else:
                acc_frame = frame_data

        return (acc_frame > 0).astype(int)
