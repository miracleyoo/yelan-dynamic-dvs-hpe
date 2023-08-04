import os
import io
import torch
import time
import argparse
import functools
import pickle as pkl
import os.path as op
from pathlib2 import Path
from datetime import datetime


def get_folder(path):
    """ Return a path to a folder, creating it if it doesn't exist 
    Args:
        path: The path of the new folder.
    Returns:
        _ : The guaranteed path of the folder/file.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_model_path(root=None, version=None, v_num=None, best=False):
    """ When best = True, return the best model's path in a directory
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the
        first three args.
    Args:
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """

    def sort_by_epoch(path):
        name = path.stem
        epoch = int(name.split('-')[1].split('=')[1])
        return epoch

    def generate_root():
        if root is not None:
            return root
        elif version is not None:
            return str(Path('lightning_logs', version, 'checkpoints'))
        else:
            return str(Path('lightning_logs', f'version_{v_num}', 'checkpoints'))

    if root == version == v_num == None:
        return None

    root = generate_root()
    if Path(root).is_file():
        return root
    if best:
        files = [i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        files.sort(key=sort_by_epoch, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root) / 'last.ckpt')
    return res


def load_model_path_by_args(args):
    return load_model_path(root=args.load_dir, version=args.load_ver, v_num=args.load_v_num, best=args.load_best)


def build_working_tree(root=None, name=''):
    if root is None:
        root = get_folder(op.join(os.getcwd(), 'lightning_logs'))

    now = datetime.now()
    dt = now.strftime("%m-%d-%H-%M-%S")
    name = f'{dt}_{name}' if len(name) != 0 else dt
    exp_path = get_folder(op.join(root, name))
    # os.chdir(exp_path)
    print(f'[√] Now working under directory: {exp_path}')
    # exp_path = os.getcwd()
    logger_dir = get_folder(op.join(exp_path, "tb_logs"))
    checkpoint_dir = get_folder(op.join(exp_path, "checkpoints"))
    recorder_dir = get_folder(op.join(exp_path, "recorder"))
    log_profiler = op.join(exp_path, "profile.txt")
    return logger_dir, checkpoint_dir, recorder_dir, log_profiler


def SBool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_gpu_num(gpus):
    if type(gpus) is str:
        gpus = int(gpus)
    elif type(gpus) in (list, tuple):
        gpus = len(gpus)
    assert type(gpus) is int
    return gpus

class Timer(object):
    """ A timer 
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.t_start = time.time()

    def __exit__(self, type, value, traceback):
        print("==> [%s]:\t" % self.name, end="")
        self.time_elapsed = time.time() - self.t_start
        print("Elapsed Time: %s (s)" % self.time_elapsed)


def tic_toc(func):
    """Print the time consumption of a certain function when a it is excuted.
    Args:
        func: An function.
    Usage:
        @tic_toc
        def func():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kw):
        tic = time.time()
        res = func(*args, **kw)
        print("==> [%s] executed in %.4f s." %
              (func.__name__, time.time() - tic))
        return res
    return wrapper

def time_string():
    """ Generate a time string from year to second.
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def pkl_load(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def pkl_dump(obj, path):
    with open(path, 'wb') as f:
        pkl.dump(obj, f)

class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def get_new_path(path):
    """ Return a path to a file, creat its parent folder if it doesn't exist, creat new one if existing.

    If the folder/file already exists, this function will use `path`_`idx` as new name, and make
    corresponding folder if `path` is a folder.
    idx will starts from 1 and keeps +1 until find a valid name without occupation.

    If the folder and its parent folders don't exist, keeps making these series of folders.

    Args:
        path: The path of a file/folder.
    Returns:
        _ : The guaranteed new path of the folder/file.
    """
    path = Path(path)
    root = Path(*path.parts[:-1])

    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        new_path = path
        if new_path.suffix == '':
            new_path.mkdir()
    else:
        idx = 1
        while True:
            stem = path.stem+"_"+str(idx)
            new_path = root / (stem+path.suffix)
            if not new_path.exists():
                if new_path.suffix == '':
                    new_path.mkdir()
                break
            idx += 1
    return str(new_path)