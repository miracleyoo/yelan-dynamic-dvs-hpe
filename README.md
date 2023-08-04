# DVS-HPE-Light

This repository is a modified and simplified version from previous [one](https://github.com/IIT-PAVIS/lifting_events_to_3d_hpe). It is built with pytorch-lightning and miracleyoo's pl [template](https://github.com/miracleyoo/pytorch-lightning-template). Hydra is not used.

Default Training Command Used: `python main.py --data_dir=<DATA_DIR> --use_mask='unet' --mask_net_path=<MASK_NET_PATH.ckpt> --batch_size=16 --gpus=8 --partial_dataset=0.3`