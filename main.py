import os
import pytorch_lightning as pl
from argparse import ArgumentParser

import torch.cuda
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from scripts.model import ModelInteface
from scripts.data import DataInterface
from scripts.utils import load_model_path_by_args, SBool, build_working_tree, get_gpu_num


def load_callbacks(checkpoint_dir=None):
    callbacks = []
    
    callbacks.append(plc.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10
    ))

    callbacks.append(plc.ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor='val_loss',
        filename='best-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        mode='min',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    
    if args.pruning is not None:
        if 'unstructured' not in args.pruning:
            callbacks.append(plc.ModelPruning(args.pruning, amount=args.pruning_amount, pruning_dim=1, pruning_norm='fro', use_global_unstructured=False, parameter_names=["weight"]))
        else:
            callbacks.append(plc.ModelPruning(args.pruning, amount=args.pruning_amount, parameter_names=["weight", "bias"]))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    logger_dir, checkpoint_dir, recorder_dir, log_profiler = build_working_tree(name='')
    args.recorder_dir = recorder_dir

    print(checkpoint_dir)
    load_path = load_model_path_by_args(args)
    print(args.data_dir)
    data_module = DataInterface(**vars(args))

    args.callbacks = load_callbacks(checkpoint_dir=checkpoint_dir)

    if load_path is None:
        model = ModelInteface(**vars(args))
    if args.load_weights_only:
        model = ModelInteface.load_from_checkpoint(load_path, **vars(args), strict=False)
        print(f'Loading weights only from checkpoint {load_path}')
    else:
        model = ModelInteface(**vars(args))

    if args.use_profiler:
        profiler = pl.profiler.AdvancedProfiler(log_profiler)
        args.profiler = profiler

    # If you want to change the logger's saving folder
    logger = TensorBoardLogger(save_dir=logger_dir, name='')
    args.logger = logger

    trainer = Trainer.from_argparse_args(args)
    if not args.test_only:
        if load_path is not None and not args.load_weights_only:
            print(f'Loading all training states from {load_path}')
            trainer.fit(model, data_module, ckpt_path=load_path)
        else:
            trainer.fit(model, data_module)
    # automatically auto-loads the best weights from the previous run
    trainer.test(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--test_only', type=SBool, default=False, nargs='?', const=True, help="Only run the test function.")
    parser.add_argument('--all_test', type=SBool, default=False, nargs='?', const=True, help="All the data files are used as test set.")

    # LR Scheduler
    parser.add_argument('--lr_scheduler', default='step', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=10, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)
    parser.add_argument('--load_weights_only', type=SBool, default=False, nargs='?', const=True)
    
    # Training Info
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    parser.add_argument('--use_profiler', type=SBool, default=False, nargs='?', const=True)
    parser.add_argument('--pruning', type=str, default=None, choices=('ln_structured', 'l1_unstructured', 'random_structured', 'random_unstructured'), help='Use pruning technique.')
    parser.add_argument('--pruning_amount', default=0.5, type=float, help='If we use pruning, what is the amount.')

    # Loss & Metrics Info
    parser.add_argument('--loss', default='MultiPixelWiseLoss', type=str, help="Loss type.", choices=('MultiPixelWiseLoss', 'PixelWiseLoss'))
    parser.add_argument('--reduction', default='mask_mean', type=str, help="Reduction method of loss.", choices=('mean', 'mask_mean', 'sum'))
    parser.add_argument('--divergence', type=SBool, default=True, nargs='?', const=True, help="Whether to use divergence in loss or not.")
    parser.add_argument('--metrics', type=str, nargs='*', default=['AUC', 'PCK', 'MPJPE'], help="metrics used in evaluation. Multiple input supported.")

    # Model Info
    parser.add_argument('--model_name', default='sad_pose', type=str, help="The main model name.")
    parser.add_argument('--hpe_backbone', default='hourglass', type=str, choices=('hourglass', 'residual', 'unet', 'bottleneck'), help="The HPE backbone of the model.")
    parser.add_argument('--use_convlstm', type=SBool, default=True, nargs='?', const=True, help="Use BiConvLSTM in the model.")
    parser.add_argument('--use_transformer', type=SBool, default=False, nargs='?', const=True, help="Use Transformer in the model to replace ConvLSTM.")
    parser.add_argument('--transformer_mask', type=SBool, default=True, nargs='?', const=True, help="Use mask in Transformer.")
    
    parser.add_argument('--stage_cl', type=SBool, default=False, nargs='?', const=True, help="Use BiConvLSTM in the model `stage` part at the tail.")
    parser.add_argument('--binary_mask', type=SBool, default=True, nargs='?', const=True, help="Turn predicted mask into binary mask before following forwarding.")
    parser.add_argument('--mask_thres', default=0.1, type=float, help="The threshold of mask binarization.")    
    parser.add_argument('--extractor_name', type=str, default='resnet34', help="The name of feature extractor.")
    parser.add_argument('--extractor_pretrained', type=SBool, default=False, nargs='?', const=True, help="Whether the feature extractor load the pretrained model.")
    parser.add_argument('--extractor_path', type=str, help="If use pretrained extractor, input its path here.")
    parser.add_argument('--input_channel_num', default=6, type=int, help="Input data channel number. 8 for NTORE.")
    parser.add_argument('--n_stages', default=3, type=int, help="The total stage number used in the model.")
    parser.add_argument('--use_mask', type=str, default='unet', choices=('none', 'unet'), help="Whether to use unet/u2net as mask predictor in the model.")
    parser.add_argument('--mask_net_path', type=str, help="The path of trained u2net, if any.")
    parser.add_argument('--log_frequency', default=100, type=int, help="Validation sample output frequency.")
    parser.add_argument('--mask_strategy', default='mask', type=str, choices=('mask', 'concat', 'mask_concat'), help="Mask combination strategy.")
    parser.add_argument('--cl_skip', type=SBool, default=True, nargs='?', const=True, help="Add skip connection to the convlstm in sad pose.")
    parser.add_argument('--ntore_2bands', type=SBool, default=False, nargs='?', const=True, help="Only use the first neg and pos band in ntore for training.")
    parser.add_argument('--time_count_baseline', type=SBool, default=False, nargs='?', const=True, help="Indicating that the experiment running now is a time count baseline.")
    parser.add_argument('--detailed_test_record', type=SBool, default=True, nargs='?', const=True, help="Whether to record the original un-averaged MPJPE values.")
    parser.add_argument('--torch_mask_net', type=SBool, default=False, nargs='?', const=True, help="Use pure pytorch unet model instead of pytorch lightning checkpoint.")
    parser.add_argument('--unfreeze_hourglass_num', default=0, type=int, choices=[0,1,2,3], help="Freeze all the other layers except last `unfreeze_hourglass_num` hourglass blocks.")

    # Dataset Info
    parser.add_argument('--dataset', default='ntore_dataset', type=str, help="The Dataset class to use.")
    parser.add_argument('--data_dir', default='ref/data', type=str)
    parser.add_argument("--seq_len", default=16, type=int, help='The sequence length used in the convlstm.')
    parser.add_argument('--n_joints', default=13, type=int, help="The total joint number used in this code.")
    parser.add_argument('--frame_size', type=int, nargs='*', default=[260,346], help="Characters whos are only used in the test session.")
    parser.add_argument('--estimate_depth', type=SBool, default=False, nargs='?', const=True, help="Does the model need to do depth estimation (when camera extrinsic is not provided).")
    parser.add_argument('--torso_length', default=1.7, type=float, help="The average torso length used when estimating the distance.")
    parser.add_argument('--base_number', default=16, type=int, help="The base number of each meta file data piece number.")
    parser.add_argument('--percentile', default=90, type=float, help="The percentile used to generate the extra band in ntore.")
    parser.add_argument("--cache_size", default=1, type=int, help="Cache size of ToreSeqReader in dataset.")
    parser.add_argument('--test_characters', type=str, nargs='*', default=['Diluc', 'Xiangling'], help="Characters whos are only used in the test session.")
    parser.add_argument('--real_test', type=SBool, default=True, nargs='?', const=True, help="If true, use the test characters for test. Elsewise, use validation set.")
    
    parser.add_argument('--remove_back_view', type=SBool, default=False, nargs='?', const=True, help="Don't use dataset that contains back view.")
    parser.add_argument('--front_only', type=SBool, default=False, nargs='?', const=True, help="Only use dataset that contains front view.")
    parser.add_argument('--remove_db', type=SBool, default=False, nargs='?', const=True, help="Don't use dataset that contains dynamic background.")
    parser.add_argument('--remove_sb', type=SBool, default=True, nargs='?', const=True, help="Don't use dataset that contains static background.")
    
    parser.add_argument('--remove_ll', type=SBool, default=False, nargs='?', const=True, help="Don't use dataset that contains low light condition.")
    parser.add_argument('--rand_test', type=SBool, default=True, nargs='?', const=True, help="Use randomly selected test cases. Only apply in SAD.")
    
    parser.add_argument('--labels_processed', type=SBool, default=False, nargs='?', const=True, help="Use the pre-processed label files.")
    parser.add_argument('--redundant_labels', type=SBool, default=False, nargs='?', const=True, help="The redundant label items which will not be used in the training, but may help in the prediction and visualization.")
    parser.add_argument('--partial_dataset', default=1.0, type=float, help="The percentage of data that is going to be use in training and validation.")
    parser.add_argument('--occlusion', type=SBool, default=False, nargs='?', const=True, help="Add occlusion when training.")
    parser.add_argument('--occlusion_test', type=SBool, default=False, nargs='?', const=True, help="Add occlusion when testing.")
    parser.add_argument('--occlusion_rate', default=0.5, type=float, help="The percentage of data that is going to be applied occlusion.")
    parser.add_argument('--use_split_file', type=SBool, default=True, nargs='?', const=True, help="Use the split file to split the dataset.")
    
    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=100)
    parser.set_defaults(strategy='dp')
    parser.set_defaults(precision=16)
    parser.set_defaults(find_unused_parameters=False)
    parser.set_defaults(gpus=4 if torch.cuda.is_available() else 0)

    args = parser.parse_args()
    args.gpus = get_gpu_num(args.gpus)

    args.ddhp = True if 'ddhp22' in args.data_dir else False
    if args.mask_net_path is not None and args.mask_net_path.endswith('.pt'):
        args.torch_mask_net = True 

    print("===== Batch Size: ", args.batch_size, "=====")
    
    if args.ntore_2bands:
        args.input_channel_num=2
    
    if args.time_count_baseline:
        args.input_channel_num = 1
        print("===== Time Count Baseline =====")
        print("[Info] Input Channel Num: ", args.input_channel_num)

    if args.mask_strategy == 'mask':
        args.input_channel_num = args.input_channel_num
    elif args.mask_strategy == 'concat':
        args.input_channel_num = args.input_channel_num + 1
    elif args.mask_strategy == 'mask_concat':
        args.input_channel_num = 2 * args.input_channel_num
    else:
        raise ValueError("Invalid mask strategy!")

    main(args)
