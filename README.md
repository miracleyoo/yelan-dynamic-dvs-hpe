# Neuromorphic high-frequency 3D dancing pose estimation in dynamic environment

- This repository is the official implementation of paper *[Neuromorphic high-frequency 3D dancing pose estimation in dynamic environment](https://www.sciencedirect.com/science/article/pii/S0925231223005118)* in *Neurocomputing, 547, 2023.*
- [Here](http://bit.ly/yelan-research) is the introduction video of the project and datasets we collected/generated. 
- The dataset is released [here](https://dataplanet.ucsd.edu/dataverse/yelan/) on UCSD DataPlanet Website.

It is built with pytorch-lightning and miracleyoo's pl [template](https://github.com/miracleyoo/pytorch-lightning-template), with the basic toolsets borrowed from [lifting_events_to_3d_hpe](https://github.com/IIT-PAVIS/lifting_events_to_3d_hpe). 

Default Training Command Used: `python main.py --data_dir=<DATA_DIR> --use_mask='unet' --mask_net_path=<MASK_NET_PATH.ckpt> --batch_size=16 --gpus=8 --partial_dataset=0.3`
