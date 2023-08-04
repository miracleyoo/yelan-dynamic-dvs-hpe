"""
Implementation of Margipose Model. Thanks to A.  Nibali --- original source
code: https://github.com/anibali/margipose/src/margipose
"""

from typing import Any, List, Tuple
import numpy as np
import torch
from torch import nn


# from .common import CLSelfAttention
from .convlstm import BiConvLSTM
from .unet import UNet_HPE
from .transformer_parts import Seq2SeqTransformer
from .resnet import Bottleneck
from ..utils import (
    FlatSoftmax,
    _down_stride_block,
    _regular_block,
    _up_stride_block,
    get_backbone_last_dimension,
    init_parameters,
)

from .unet_interface import ModelInteface as Unet

class MargiPoseStage(nn.Module):
    MIN_FEATURE_DIMENSION = 64
    MAX_FEATURE_DIMENSION = 128

    def __init__(self, n_joints, mid_shape, heatmap_space, hpe_backbone='hourglass'):
        super().__init__()

        self.n_joints = n_joints
        self.softmax = FlatSoftmax()
        self.heatmap_space = heatmap_space
        self.hpe_backbone = hpe_backbone
        mid_feature_dimension = mid_shape[0]
        min_dimension = MargiPoseStage.MIN_FEATURE_DIMENSION
        max_dimension = MargiPoseStage.MAX_FEATURE_DIMENSION

        if hpe_backbone == 'hourglass':
            up_padding = (
                int(mid_shape[1] % 2 != 1),
                int(mid_shape[2] % 2 != 1),
            )  # TODO: eval up_padding basin on mid_shape

            self.down_layers = nn.Sequential(
                _regular_block(mid_feature_dimension, min_dimension),
                _regular_block(min_dimension, min_dimension),
                _down_stride_block(min_dimension, max_dimension),
                _regular_block(max_dimension, max_dimension),
                _regular_block(max_dimension, max_dimension),
            )
            self.up_layers = nn.Sequential(
                _regular_block(max_dimension, max_dimension),
                _regular_block(max_dimension, max_dimension),
                _up_stride_block(max_dimension, min_dimension, padding=up_padding),
                _regular_block(min_dimension, min_dimension),
                _regular_block(min_dimension, self.n_joints),
            )
        elif hpe_backbone == 'unet':
            self.unet = UNet_HPE(in_ch=mid_feature_dimension, out_ch=self.n_joints, min_dimension=min_dimension, max_dimension=max_dimension)
        elif hpe_backbone == 'residual':
            self.residuals = nn.Sequential(
                _regular_block(mid_feature_dimension, min_dimension),
                _regular_block(min_dimension, min_dimension),
                _regular_block(min_dimension, max_dimension),
                _regular_block(max_dimension, max_dimension),
                _regular_block(max_dimension, max_dimension),
                _regular_block(max_dimension, max_dimension),
                _regular_block(max_dimension, max_dimension),
                _regular_block(max_dimension, min_dimension),
                _regular_block(min_dimension, min_dimension),
                _regular_block(min_dimension, self.n_joints),
            )
        elif hpe_backbone == 'bottleneck':
            self.bottlenecks = nn.Sequential(
                Bottleneck(mid_feature_dimension, min_dimension),
                Bottleneck(min_dimension, min_dimension),
                Bottleneck(min_dimension, max_dimension),
                Bottleneck(max_dimension, max_dimension),
                Bottleneck(max_dimension, max_dimension),
                Bottleneck(max_dimension, max_dimension),
                Bottleneck(max_dimension, max_dimension),
                Bottleneck(max_dimension, min_dimension),
                Bottleneck(min_dimension, min_dimension),
                Bottleneck(min_dimension, self.n_joints),
            )
        else:
            raise ValueError(f'Unknown hpe_backbone: {hpe_backbone}')

        init_parameters(self)

    def forward(self, *inputs):
        if self.hpe_backbone == 'hourglass':
            mid_in = self.down_layers(inputs[0])
            res = self.up_layers(mid_in)
        elif self.hpe_backbone == 'unet':
            res = self.unet(inputs[0])
        elif self.hpe_backbone == 'residual':
            res = self.residuals(inputs[0])
        elif self.hpe_backbone == 'bottleneck':
            res = self.bottlenecks(inputs[0])
        return res


class SadPose(nn.Module):
    """
    Multi-stage marginal heatmap estimator
    """
    def __init__(self, n_stages, 
                 in_cnn, 
                 frame_size, 
                 input_channel_num,
                 n_joints, 
                 use_mask, 
                 mask_net_path, 
                 binary_mask, 
                 mask_thres=0.1, 
                 mask_strategy='mask', 
                 use_convlstm=True, 
                 cl_skip=False, 
                 ntore_2bands=False, 
                 torch_mask_net=False, 
                 unfreeze_hourglass_num=0,
                 use_transformer=False,
                 transformer_mask=False,
                 hpe_backbone='hourglass',
                 **kwargs):
        super().__init__()
        print('Model init entered!')
        self.n_stages = n_stages
        self.use_convlstm = use_convlstm
        self.cl_skip = cl_skip
        self.ntore_2bands = ntore_2bands
        self.use_mask = use_mask
        self.in_cnn = in_cnn  # Backbone provided as parameter
        self.mask_net_path = mask_net_path
        self.mask_thres = mask_thres
        self.binary_mask = binary_mask
        self.mask_strategy = mask_strategy.lower()
        self.input_channel_num = input_channel_num
        self.torch_mask_net = torch_mask_net            
        self.unfreeze_hourglass_num = unfreeze_hourglass_num
        self.use_transformer = use_transformer
        self.transformer_mask = transformer_mask
        self.hpe_backbone = hpe_backbone

        self.in_shape = (self.input_channel_num, *frame_size)
        self.mid_shape = get_backbone_last_dimension(in_cnn, self.in_shape)
        self.mid_feature_dimension = self.mid_shape[0]

        self.xy_hm_cnns = nn.ModuleList()
        self.zy_hm_cnns = nn.ModuleList()
        self.xz_hm_cnns = nn.ModuleList()
        self.hm_combiners = nn.ModuleList()
        self.softmax = FlatSoftmax()

        self.n_joints = n_joints
        self._set_stages()
        print(f'[√] Using {self.hpe_backbone} as backbone for HPE.')
        
        if self.use_transformer:
            print("[√] Using Transformer in SadPose. (Instead of ConvLSTM)")
            self.adp_pool = nn.AdaptiveAvgPool2d((1, 1))
            # Define the transformer model
            self.transformer = Seq2SeqTransformer(
                d_model=self.mid_feature_dimension,
                nhead=8,
                num_layers=6,
                dim_feedforward=2048,
                dropout=0.1,
                use_mask=self.transformer_mask,
            )
        elif self.use_convlstm:
            print("[√] Using ConvLSTM in SadPose.")
            self.cl = BiConvLSTM(self.mid_feature_dimension, self.mid_feature_dimension,
                             (3, 3), 1, batch_first=True)      
            if self.cl_skip:
                self.cl_skip_block = _regular_block(in_chans=self.mid_feature_dimension*2, out_chans=self.mid_feature_dimension)

        self.unfreeze_hourglass_blocks(unfreeze_hourglass_num=self.unfreeze_hourglass_num)

        if self.use_mask == 'unet':
            if self.torch_mask_net:
                self.mask_net = torch.jit.load(self.mask_net_path)
                print('[Info] Using pure pytorch version unet...')
            else:
                print('Loading mask net from ckpt!')
                self.mask_net = Unet.load_from_checkpoint(self.mask_net_path).model
            self.mask_net.eval()
            for param in self.mask_net.parameters():
                param.requires_grad = False
            print("[√] UNet Freezed.")
        else:
            self.mask_net = None
            print("[x] Not using any masking network...")
        
        

    class HeatmapCombiner(nn.Module):
        def __init__(self, n_joints, n_planes, out_channels):
            super().__init__()
            self.combine_block = _regular_block(n_joints * n_planes, out_channels)

        def forward(self, x):
            return self.combine_block(x)

    def _set_stages(self):
        for t in range(self.n_stages):
            if t > 0:
                self.hm_combiners.append(
                    SadPose.HeatmapCombiner(
                        self.n_joints, 3, self.mid_feature_dimension
                    )
                )
            self.xy_hm_cnns.append(
                MargiPoseStage(
                    self.n_joints,
                    self.mid_shape,
                    heatmap_space='xy',
                    hpe_backbone=self.hpe_backbone,
                )
            )
            self.zy_hm_cnns.append(
                MargiPoseStage(
                    self.n_joints,
                    self.mid_shape,
                    heatmap_space='zy',
                    hpe_backbone=self.hpe_backbone,
                )
            )
            self.xz_hm_cnns.append(
                MargiPoseStage(
                    self.n_joints,
                    self.mid_shape,
                    heatmap_space='xz',
                    hpe_backbone=self.hpe_backbone,
                )
            )

    def forward(self, inputs, mask_input=None) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        Model forward process
        Args:
            inputs: input batch

        Returns:
            Triple of list of heatmaps of length {n_stages}
        """
        B, S, C, H, W = inputs.shape
        inputs = inputs.reshape((B*S, C, H, W))

        if self.use_mask != 'none':
            x0 = inputs
            if mask_input is None:
                if self.torch_mask_net:
                    mask_ori = predict_mask(self.mask_net, x0)
                else:
                    mask_ori = self.mask_net.predict_mask(x0)
            else:
                mask_ori = mask_input

            if self.binary_mask:
                mask = torch.where(mask_ori>self.mask_thres,1,0)
            else:
                mask = mask_ori
            if self.mask_strategy == 'mask':
                inputs = mask * inputs
            elif self.mask_strategy == 'concat':
                inputs = torch.concat((inputs, mask), axis=1)
            elif self.mask_strategy == 'mask_concat':
                masked = mask * inputs
                inputs = torch.concat((inputs, masked), axis=1)

        if self.ntore_2bands:
            inputs = inputs[:,np.r_[0,3]]

        features = self.in_cnn(inputs)

        if self.use_transformer:
            channel_weight = self.adp_pool(features).squeeze().reshape(B, S, -1).permute(1, 0, 2) # (S, B, C)
            channel_weight = self.transformer(channel_weight) # (S, B, C)
            channel_weight = channel_weight.permute(1, 0, 2).reshape(B*S, -1, 1, 1) # (B*S, C, 1, 1)
            features = features * channel_weight
            
        elif self.use_convlstm:
            cl_out = self.cl(features.reshape(B, S, *features.shape[1:]))[0]
            cl_out = cl_out.reshape(B*S, *cl_out.shape[2:])
            if self.cl_skip:
                features = torch.concat((features, cl_out), axis=1)
                features = self.cl_skip_block(features)
            else:
                features = cl_out

        xy_heatmaps: List[Any] = []
        zy_heatmaps: List[Any] = []
        xz_heatmaps: List[Any] = []

        inp = features

        for t in range(self.n_stages):
            if t > 0:
                combined_hm_features = self.hm_combiners[t - 1](
                    torch.cat(
                        [xy_heatmaps[t - 1], zy_heatmaps[t - 1], xz_heatmaps[t - 1]], -3
                    )
                )
                inp = inp + combined_hm_features

            xy_heatmaps.append(self.softmax(self.xy_hm_cnns[t](inp)))
            zy_heatmaps.append(self.softmax(self.zy_hm_cnns[t](inp)))
            xz_heatmaps.append(self.softmax(self.xz_hm_cnns[t](inp)))
        if self.use_mask != 'none':
            return xy_heatmaps, zy_heatmaps, xz_heatmaps, mask_ori
        else:
            return xy_heatmaps, zy_heatmaps, xz_heatmaps

    def unfreeze_hourglass_blocks(self, unfreeze_hourglass_num=0):
        if unfreeze_hourglass_num > 0:
            for para in self.parameters():
                para.requires_grad = False
            retrain_layers = ['xy_hm_cnns.{:d}'.format(i) for i in range(self.n_stages-1, max(self.n_stages-1-unfreeze_hourglass_num, -1), -1)] + \
                            ['zy_hm_cnns.{:d}'.format(i) for i in range(self.n_stages-1, max(self.n_stages-1-unfreeze_hourglass_num, -1), -1)] + \
                            ['xz_hm_cnns.{:d}'.format(i) for i in range(self.n_stages-1, max(self.n_stages-1-unfreeze_hourglass_num, -1), -1)] + \
                            ['hm_combiners.{:d}'.format(i) for i in range(self.n_stages-2, max(self.n_stages-2-unfreeze_hourglass_num, -1), -1)]
            params = self.state_dict()
            filtered_names = [i for i in params.keys() if any(j in i for j in retrain_layers)]
            for name, param in self.named_parameters():
                if name in filtered_names:
                    param.requires_grad = True
            print("@@@All parameters are frozen except last {:d} hourglass blocks".format(unfreeze_hourglass_num))

def get_sadpose_model(params):
    return SadPose(**params)

def predict_mask(unet, x):
    x1 = unet.inc(x)
    x2 = unet.down1(x1)
    x3 = unet.down2(x2)
    x4 = unet.down3(x3)
    x5 = unet.down4(x4)

    x = unet.up1(x5, x4)
    x = unet.up2(x, x3)
    x = unet.up3(x, x2)
    x = unet.up4(x, x1)

    masks = unet.outc(x).squeeze()
    masks = torch.sigmoid(masks)[:,0:1]
    return masks