import os
import cv2
import sys
import glob
import copy
import time
import torch
import platform

import os.path as op
import numpy as np
from dotdict import dotdict
from matplotlib import pyplot as plt
import pathlib2

sys.path.append('..')
from scripts.model import ModelInteface
from scripts.utils import *
from scripts.model.metrics import MPJPE
from scripts.utils.skeleton_helpers import Skeleton

device = 'cuda'
frame_size = (260, 346)
mpjpe = MPJPE()
nidhp_root = r'L:\DVS\models\hpe\NiDHP'
midhp_root = r'L:\DVS\models\hpe\MiDHP'
mask_net_path = r'L:\DVS\models\unet\05-19-18-28-41\checkpoints\best-epoch=09-val_loss=0.021.ckpt'
model_date_dict = {
    'MiDHP': {
        'w_cl':'05-25-16-19-04',
        'const_time':'05-19-23-41-59',
        'occlusion': '10-17-21-15-12'
        },
    'NiDHP': {
        'w_cl':'06-01-16-13-56',
        'const_time':'06-01-16-17-07',
        'pretrain':'06-02-00-34-40'
        }
}


def platform_check():
    # Get Basic Running Info
    print('Current Directory: ', os.getcwd())
    print('Current Node: ', platform.node())
    print('Cuda availability: ', torch.cuda.is_available())

    if platform.system() == 'Windows':
        temp = pathlib2.PosixPath
        pathlib2.PosixPath = pathlib2.WindowsPath

def time_string():
    """ Generate a time string from year to second.
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def get_model_path(date, root=midhp_root, best=True):
    model_subdir = op.join(root, date, 'checkpoints')
    if best:
        model_path = glob.glob(op.join(model_subdir, 'best*'))[0]
    else:
        model_path = op.join(model_subdir, 'last.ckpt')
    return model_path

def load_model(time_str='05-25-16-19-04', best=True, device='cuda', root=midhp_root):
    model_path = get_model_path(time_str, root=root, best=True)
    print(model_path)
    model = ModelInteface.load_from_checkpoint(model_path, mask_net_path=mask_net_path, detailed_test_record=True)
    model = model.to(device)
    print('Using trained model: ', model_path)
    return model

def manual_load_data(data_path, idx=0):
    reader = ToreSeqReader([data_path], labels_processed=False, percentile=90, redundant_labels=True)
    print("Total tore number:", reader.metas[0]['total_tore_count'])

    b_x, b_y = reader.load_seq_by_index(0, idx)
    tores = torch.tensor(b_x).float().to(device)
    del b_y['name']
    for k in b_y.keys():
        b_y[k] = torch.tensor(b_y[k])
    return b_x, b_y, tores

def inference(batch, model, verbose=False):
    b_x, b_y = batch
    del b_y['name']
    for k in b_y.keys():
        b_y[k] = torch.tensor(b_y[k])
    b_x = torch.tensor(b_x).float()
    normed_pred_joints, masks = model.predict_step(b_x.unsqueeze(0).to(device))
    normed_pred_joints = normed_pred_joints.cpu().detach()
    del batch

    if verbose:
        print('Label Keys: ', b_y.keys())
        print('Extrinsic:\n', b_y['M'][0].numpy().round(2))
        print('Intrinsic:\n', b_y['camera'][0].numpy().round(2))
    return normed_pred_joints, masks, b_x, b_y

suf = lambda n: "%d%s"%(n,{1:"st",2:"nd",3:"rd"}.get(n if n<20 else n%10,"th"))


def get_case_data_reader(case_root, case_name, *args, **kargs):
    data_path = op.join(case_root, case_name, 'meta.json')
    reader = ToreSeqReader([data_path], *args, **kargs)
    print('TORE Reader Initialized!')
    return reader

def build_rec(b_x, normed_pred_joints, masks, model, b_y=None, gt_mask=None, intrinsic_matrix=None, extrinsic_matrix=None):
    tore_vis = np.array([ntore_to_redgreen(b_x[i]) for i in range(16)])

    rec = dotdict({
        'b_x': b_x,
        'tore_vis': tore_vis,
        'pred_mask': masks.cpu().detach().squeeze().numpy(),
        'bn_mask': torch.where(masks>0.1,1,0).cpu().detach().squeeze().numpy(),
        'pred_sknormed': normed_pred_joints.numpy()
    })

    if b_y is not None:
        pred_skcam = model.denormalize_predictions(normalized_predictions=normed_pred_joints, b_y=b_y)
        pred_skori = np.array([reproject_xyz_onto_world_coord(pred_skcam[i], b_y['M'][i]).numpy() for i in range(16)])
        rec['intrinsic_matrix'] = b_y['camera'][0]
        rec['extrinsic_matrix'] = b_y['M'][0]
        rec['gt_skcam'] = b_y['skeleton'].numpy()
        rec['gt_sknormed'] = b_y['normalized_skeleton'].numpy()
        rec['gt_skori'] = b_y['xyz'].numpy()
        rec['pred_skcam'] = pred_skcam.numpy(),
        rec['pred_skori'] = pred_skori

    if intrinsic_matrix is not None:
        rec['intrinsic_matrix'] = intrinsic_matrix

    if extrinsic_matrix is not None:
        rec['extrinsic_matrix'] = extrinsic_matrix

    if gt_mask is not None:
        rec['gt_mask'] = gt_mask
    return rec


def translate_3d_to_2d_joints(joints, intrinsic_matrix, extrinsic_matrix, w=346, h=260):
    sk = Skeleton(joints)
    joints_2d = torch.tensor(sk.get_2d_points(
        w,
        h,
        extrinsic_matrix=extrinsic_matrix,
        intrinsic_matrix=intrinsic_matrix,
    ))
    joints = torch.stack([joints_2d[:, 0], joints_2d[:, 1]], 1)
    return joints


def vis_center(rec, inbatch_idx, vis_config={'overlay_comp':True}, ret_fig=False):
    keys = vis_config.keys()
    figs_dict = {}
    if 'overlay_comp' in keys and vis_config['overlay_comp']:
        fig = plot_2d_overlay(rec['gt_skori'][inbatch_idx], 
                            rec['intrinsic_matrix'], 
                            rec['extrinsic_matrix'], 
                            rec['tore_vis'][inbatch_idx][0].transpose(1,2,0),
                            frame_size=(260, 346), 
                            plot_lines=True,
                            ret_fig=True, 
                            pred_pose=rec['pred_skori'][inbatch_idx])
        figs_dict['overlay_comp'] = fig
        
    if 'overlay_pred' in keys and vis_config['overlay_pred']:
        fig = plot_2d_overlay(rec['pred_skori'][inbatch_idx], 
                            rec['intrinsic_matrix'], 
                            rec['extrinsic_matrix'], 
                            rec['tore_vis'][inbatch_idx][0].transpose(1,2,0),
                            frame_size=(260, 346), 
                            plot_lines=True,
                            ret_fig=True)
        figs_dict['overlay_pred'] = fig

    if 'mask_pred' in keys and vis_config['mask_pred']:
        fig = plt.figure(figsize=(7.05,5.2875))
        plt.imshow(rec['pred_mask'][inbatch_idx], cmap='gray')
        plt.axis('off')
        plt.show()
        figs_dict['mask_pred'] = fig

    if 'mask_binary' in keys and vis_config['mask_binary']:
        fig = plt.figure(figsize=(7.05,5.2875))
        plt.imshow(rec['bn_mask'][inbatch_idx], cmap='gray')
        plt.axis('off')
        plt.show()
        figs_dict['mask_binary'] = fig

    if 'mask_gt' in keys and vis_config['mask_gt']:
        fig = plt.figure(figsize=(7.05,5.2875))
        plt.imshow(rec['gt_mask'][inbatch_idx], cmap='gray')
        plt.axis('off')
        plt.show()
        figs_dict['mask_gt'] = fig

    if 'tore_bands' in keys and vis_config['tore_bands']:
        fig = check_ntore(rec['b_x'][inbatch_idx], ret_fig=True)
        figs_dict['tore_bands'] = fig

    if 'pred_ori_3d' in keys and vis_config['pred_ori_3d']:
        fig = plot_skeleton_3d(rec['pred_skori'][inbatch_idx], limits = [[-1, 1], [-1, 1], [0, 2]], ret_fig=True)
        figs_dict['pred_ori_3d'] = fig

    if 'gt_ori_3d' in keys and vis_config['gt_ori_3d']:
        fig = plot_skeleton_3d(rec['gt_skori'][inbatch_idx], limits = [[-1, 1], [-1, 1], [0, 2]], ret_fig=True)
        figs_dict['gt_ori_3d'] = fig

    if 'pred_normed_3d' in keys and vis_config['pred_normed_3d']:
        fig = plot_skeleton_3d(rec['pred_sknormed'][inbatch_idx], limits = [[-1, 1], [-1, 1], [0, 2]], ret_fig=True)
        figs_dict['pred_normed_3d'] = fig

    if 'gt_normed_3d' in keys and vis_config['gt_normed_3d']:
        fig = plot_skeleton_3d(rec['gt_sknormed'][inbatch_idx], limits = [[-1, 1], [-1, 1], [0, 2]], ret_fig=True)
        figs_dict['gt_normed_3d'] = fig

    if 'full_row' in keys and vis_config['full_row']:
        combined_ntore = rec.tore_vis[inbatch_idx]
        fig_row = [combined_ntore[0]]
        pred_mask_vis = rec.pred_mask[inbatch_idx][np.newaxis,...].repeat(3,axis=0).astype(float)
        fig_row.append(pred_mask_vis)
        bn_mask_vis = rec.bn_mask[inbatch_idx][np.newaxis,...].repeat(3,axis=0).astype(float)
        fig_row.append(bn_mask_vis)
        if 'gt_mask' in rec.keys():
            gt_mask_vis = rec.gt_mask[inbatch_idx][np.newaxis,...].repeat(3,axis=0).astype(float)
            fig_row.append(gt_mask_vis)
        masked_tore_vis = combined_ntore[0]*rec.bn_mask[inbatch_idx].astype(float)
        fig_row.append(masked_tore_vis)
        ones_frame = np.ones_like(masked_tore_vis)*255
        overlay_fig = plot_2d_overlay(rec['gt_skori'][inbatch_idx], 
                                      rec['intrinsic_matrix'], 
                                      rec['extrinsic_matrix'],
                                      ones_frame.transpose(1,2,0), 
                                      frame_size=(260, 346), 
                                      pred_pose=rec.pred_skori[inbatch_idx], 
                                      plot_lines=True, 
                                      ret_fig=True, 
                                      show=False)

        overlay_vis=get_img_from_fig(overlay_fig).transpose(2,0,1).astype(float)/255
        fig_row.append(overlay_vis)
        fig_row = [f.transpose(1,2,0) for f in fig_row]
        _, newh, neww = overlay_vis.shape

        for j, f in enumerate(fig_row[:-1]):
            fig_row[j] = np.clip(cv2.resize(f, dsize=(neww, newh), interpolation=cv2.INTER_CUBIC), 0, 1)

        fig = batch_show(fig_row, ret_fig=True, sub_size=(4,3.5))
        figs_dict['full_row'] = fig
    
    if ret_fig:
        return figs_dict

def write_plt_fig(fig, root=None, path=None, prefix='', suffix='', dpi=300, pad_inches=0.01, ts=None):
    if root is not None:
        ts = ts or time_string()
        name = f'{prefix}_{time_string()}_{suffix}'.strip('_')
        path = op.join(root, f'{name}.png')
    fig = get_img_from_fig(fig, dpi=dpi, pad_inches=pad_inches)
    cv2.imwrite(path, cv2.cvtColor(fig, cv2.COLOR_BGR2RGB))

def write_all_figs(figs_dict, root=None, prefix='', dpi=300, pad_inches=0.01):
    ts = time_string()
    for k, v in figs_dict.items():
        write_plt_fig(v, root=root, prefix=prefix, suffix=k, dpi=dpi, pad_inches=pad_inches, ts=ts)


# Generate the masked version of the input
def get_masked_img(inputs, top, bottom, left, right):
    # assert len(inputs.shape) == 3
    assert 0 <= top < inputs.shape[-2]
    assert 0 <= bottom < inputs.shape[-2]
    assert 0 <= left < inputs.shape[-1]
    assert 0 <= right < inputs.shape[-1]
    masked_input = copy.deepcopy(inputs)
    masked_input[..., top:bottom, left:right] = 0
    return masked_input