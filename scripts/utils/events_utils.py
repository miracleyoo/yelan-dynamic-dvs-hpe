""" Utils for loading events.
"""
import h5py
import numpy as np
import pandas as pd
import os.path as op
import plotly.express as px
from dv import AedatFile
from math import floor, ceil
from tqdm import trange
from .vis_utils import *
from .utils import get_new_path, pkl_load

cam_settings = {
    "xz":  {'up': {'x': 0.029205246220929418, 'y': -0.07454904061931165, 'z': -0.9967895937136961}, 'center': {'x': 0.4152144677376415, 'y': -0.19700200366278003, 'z': 0.1318296812311048}, 'eye': {'x': -0.05808189772173178, 'y': 1.7511480688146275, 'z': -0.027738051796443258}},
    "side": {'up': {'x': 0.18044721455186086, 'y': -0.0326062061218738, 'z': -0.9830440672130688}, 'center': {'x': 0.4282785144271674, 'y': -0.17502657663951424, 'z': 0.21871482833583408}, 'eye': {'x': -1.1257276557476024, 'y': 1.3147917910060438, 'z': -0.11595318966741139}},
    "full": {'up': {'x': 0.16192919505818526, 'y': 0.014526753698953593, 'z': -0.9866954490696601}, 'center': {'x': 0.4282785144271674, 'y': -0.17502657663951424, 'z': 0.21871482833583408}, 'eye': {'x': -1.5964852994314518, 'y': 1.225895055808503, 'z': -0.09294925336730195}}
}


def extract_aedat4(path):
    """ Extract events from AEDAT4 file.
        Args:
            path: str, the path of input aedat4 file.
        Returns:
            events: pd.DataFrame, pandas data frame containing events.
    """
    with AedatFile(path) as f:
        events = np.hstack([packet for packet in f['events'].numpy()])
    events = pd.DataFrame(events)[['timestamp', 'x', 'y', 'polarity']]
    events = events.rename(columns={'timestamp': 't', 'polarity': 'p'})
    return events


def load_events(path, slice=None, to_df=True, start0=False, verbose=False):
    """ Load the DVS events in .h5 or .aedat4 format.
    Args:
        path: str, input file name.
        slice: tuple/list, two elements, event stream slice start and end.
        to_df: whether turn the event stream into a pandas dataframe and return.
        start0: set the first event's timestamp to 0.
    """
    ext = op.splitext(path)[1]
    assert ext in ['.h5', '.aedat4']
    if ext == '.h5':
        f_in = h5py.File(path, 'r')
        events = f_in.get('events')[:]
    else:
        events = extract_aedat4(path)
        events = events.to_numpy()  # .astype(np.uint32)

    if verbose:
        print(events.shape)
    if slice is not None:
        events = events[slice[0]:slice[1]]
    if start0:
        events[:, 0] -= events[0, 0]  # Set the first event timestamp to 0
        # events[:,2] = 260-events[:,2] # Y originally is upside down
    if to_df:
        events = pd.DataFrame(events, columns=['t', 'x', 'y', 'p'])
    return events


def plot_events_3d(events: pd.DataFrame, cam_setting=None):
    """ Visualize events in 3D space.
        Used for better understanding the sample events. 
        !!! Don't input too long event streams.
        The aspect ratio for x, y, t is 1:1:2.
        Args:
            events: pd.DataFrame.
            cam_setting: camera settings used in plotly. Used to set the camera's 
                position and view angle to the correct place.
    """
    if cam_setting is None:
        cam_setting = cam_settings['full']
    fig = px.scatter_3d(events, x='x', y='t', z='y', color='p', width=1000, height=600)
    fig.update_traces(marker={'size': 1, 'opacity': 0.25})
    fig.layout.scene = {'camera': cam_setting, 'aspectratio': {'x': 1, 'y': 2, 'z': 1}}
    fig.show()
    fig.write_image(get_new_path('results/events_vis.png'), scale=2)
    return fig


def accumulate_frame(events, frame_size=(260, 346)):
    """ Accumulate input events to a frame using the same way of DHP19
        Args: 
            events: np.array.
            frame_size: (H, W)
    """
    h, w = frame_size
    img = np.zeros((w, h))
    for event in events:
        timestamp, x, y, p = event
        img[x, y] += 1

    # Normalize
    sum_img = np.sum(img)
    count_img = np.sum(img > 0)
    mean_img = sum_img / count_img
    var_img = np.var(img[img > 0])
    sig_img = np.sqrt(var_img)

    if sig_img < 0.1/255:
        sig_img = 0.1/255

    num_sdevs = 3.0
    mean_grey = 0
    ranges = num_sdevs * sig_img
    half_range = 0
    range_new = 255

    def norm_img(z):
        if z == 0:
            res = mean_grey
        else:
            res = np.floor(np.clip((z+half_range) * range_new / ranges, 0, range_new))
        return res

    norm_img = np.vectorize(norm_img)
    img = norm_img(img)
    return img.T


def process_data_case(exp_name, camera_view, pose_name,
                      dvs_root, pose_root, camera_root,
                      out_root, events_per_frame=7500):
    """ Process the camera, pose, and events to constant event count representation.
        Args:
            exp_name: case name.
            camera_view: camera view file.
            pose_name: pose name.
            dvs_root: events root folder.
            pose_root: poses root folder.
            camera_root: camera root folder.
            out_root: output folder.
            events_per_frame: fixed event count number.
    """
    # Set paths
    out_root_frames = op.join(out_root, 'frames')
    out_root_labels = op.join(out_root, 'labels')
    dvs_path = op.join(dvs_root, exp_name, camera_view, 'events.h5')
    pose_path = op.join(pose_root, pose_name, 'motion_dict_kp13.pkl')
    camera_path = op.join(camera_root, 'front_50_camera_matrix_dvs346.pkl')

    # Load files
    dvs = h5py.File(dvs_path, 'r')
    pose = pkl_load(pose_path)['data']
    cm = pkl_load(camera_path)

    # Intrinsic Matrix
    intrinsic = np.pad(cm['K'], ((0, 0), (0, 1)), 'constant')
    # Extrinsic Matrix
    extrinsic = cm['RT']

    events = dvs['events']
    events_num = len(events)

    packet_num = events_num // events_per_frame
    # Generate files
    for event_idx in trange(packet_num, desc=f'{exp_name}_{camera_view}'):
        labels = {}
        out_stem = f'{exp_name}_frame_{event_idx}_{camera_view}'
        frame_path = op.join(out_root_frames, out_stem+'.npy')
        label_path = op.join(out_root_labels, out_stem+'.npz')

        start_idx = events_per_frame * event_idx
        end_idx = start_idx + events_per_frame
        packet_events = events[start_idx: end_idx]
        packet_frame = accumulate_frame(packet_events)

        start_time = packet_events[0][0]*1e-6
        end_time = packet_events[-1][0]*1e-6

        packet_poses = pose[int(floor(start_time*300)): int(ceil(end_time*300))]
        packet_poses_mean = packet_poses.mean(axis=0).T

        labels['camera'] = intrinsic
        labels['M'] = extrinsic
        labels['xyz'] = packet_poses_mean

        with open(frame_path, 'wb') as f:
            np.save(f, packet_frame)
        with open(label_path, 'wb') as f:
            np.savez(f, **labels)
    return exp_name, camera_view

# def fields_view(arr, fields):
#     dtype2 = np.dtype({name:arr.dtype.fields[name] for name in fields})
#     return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)
