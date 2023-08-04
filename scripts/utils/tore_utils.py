import torch
import numpy as np
import matplotlib.pyplot as plt

def gen_tore_plus(tore, threshold=None, percentile=90, split=False):
    """ Generate the PLUS version of tore volume.
        Two filtered most recent cache layers are added to the volume.
    Args:
        tore: ndarry, (6, h, w).
        percentile: float, 0~100. The percentile threshold. Works on
            neg and pose separately. Percentile has higher priority.
            80 is recommended as default percentile.
        threshold: float, 0~1. The fixed threshold for both neg and pos.
            0.5 is recommended as default threshold.

    Return:
        ndarry, (8, h, w). NTORE volume proposed.
    """
    if percentile is not None:
        # print(tore[0].shape, tore[0].mean(), tore[0].max(), tore[0].min())
        pos_selected = tore[0][tore[0] != 0]
        neg_selected = tore[3][tore[3] != 0]
        if pos_selected.shape[0] !=0:
            pos_thres = np.percentile(pos_selected, percentile)
            pos_recent = np.where(tore[0] > pos_thres, tore[0], 0)[np.newaxis, :]
        else:
            pos_recent = np.zeros_like(tore[0])[np.newaxis, :]
        if neg_selected.shape[0] !=0:
            neg_thres = np.percentile(neg_selected, percentile)        
            neg_recent = np.where(tore[3] > neg_thres, tore[3], 0)[np.newaxis, :]
        else:
            neg_recent = np.zeros_like(tore[3])[np.newaxis, :]
    elif threshold is not None:
        pos_recent = np.where(tore[0] > threshold, tore[0], 0)[np.newaxis, :]
        neg_recent = np.where(tore[3] > threshold, tore[3], 0)[np.newaxis, :]
    else:
        raise ValueError('Please specify the value of threshold or percentile!')
    tore_plus = np.concatenate((pos_recent, tore[0:3], neg_recent, tore[3:]), axis=0)
    if split:
        tore_plus[1] = tore_plus[1] - tore_plus[0]
        tore_plus[5] = tore_plus[5] - tore_plus[4]
        
    return tore_plus

def gen_tore_plus_batch(tore, threshold=None, percentile=90, split=False):
    """ Generate NTORE in batch.
    """
    assert len(tore.shape)==4
    assert tore.shape[1]==6
    
    if type(tore) == torch.Tensor:
        x = tore.cpu().detach().numpy()
    else:
        x = tore

    out = np.zeros((x.shape[0], 8 , *x.shape[2:]))
    for i in range(x.shape[0]):
        out[i] = gen_tore_plus(x[i], threshold=threshold, percentile=percentile, split=split)
    
    if type(tore) == torch.Tensor:
        out = torch.tensor(out).to(tore)
    return out

def to_float(x):
    if isinstance(x, np.ndarray):
        return x.astype(float)
    else:
        return x.float()

def ntore_to_redgreen(tore):
    """ Make each layer of tore a rgb image in which red is the positive channel and green is negative.
    Args:
        tore: ndarray, whether 6 or 8 channels.
    Return:
        combined: the combined images.
    """
    _, h, w = tore.shape
    tore = tore.reshape(2, -1, h, w)
    combined = np.concatenate((tore, np.zeros((1, tore.shape[1],h,w))), axis=0).swapaxes(0,1)
    return combined

def check_ntore(tore, ret_fig=False):
    suf = lambda n: "%d%s"%(n,{1:"st",2:"nd",3:"rd"}.get(n if n<20 else n%10,"th"))
    combined_ntore = ntore_to_redgreen(tore).transpose(0,2,3,1)
    subfig_num = combined_ntore.shape[0]
    fig = plt.figure(figsize=(subfig_num*4,4))
    for i in range(1,subfig_num+1):
        plt.subplot(1, subfig_num, i)
        plt.imshow(combined_ntore[i-1])
        plt.title(f'{suf(i)} channel')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    if ret_fig:
        return fig