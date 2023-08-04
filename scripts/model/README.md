# Model Description

- The main model we use in this project is `sad_pose.py`, and the trainer as well as the model enter point is `model_interface.py`.

- `convlstm.py`, `resnet.py`, `unet_parts.py`, `unet.py`, `transformer_parts.py` provide necessary custome modules used in our main model.

- `metrics.py` holds the metrics classes used in the evalueation, while `losses.py` contains losses applied during training.

- `unet_interface.py` is a file for training the stage1 UNet, and here it is mainly used for loading the trained UNet.