import importlib
import inspect
import os.path as op
import pickle as pkl

import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as lrs

from ..utils import Skeleton, average_loss, get_feature_extractor
from .losses import MultiPixelWiseLoss, PixelWiseLoss, predict3d
from .metrics import AUC, MPJPE, PCK

class ModelInteface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kwargs):
        super().__init__()
        print('Entering ModelInteface...')
        self.save_hyperparameters(ignore=['in_cnn'])
        if 'callbacks' in self.hparams.keys():
            del self.hparams['callbacks']
        self.load_feature_extractor()
        self.load_model()
        self.configure_loss()
        self.configure_metrics()
        self.eval_counter = 0
        self.atomized_logs = []

        if self.hparams.detailed_test_record:
            self.ori_mpjpe = MPJPE()
            print('Adding original mpjpe values to results.')
        print('Model Initialized.')
    
    def forward(self, x, mask_input=None):
        """
        For inference. Return normalized skeletons
        """
        outs = self.model(x, mask_input=mask_input)
        xy_hm = outs[0][-1]
        zy_hm = outs[1][-1]
        xz_hm = outs[2][-1]

        pred_joints = predict3d(xy_hm, zy_hm, xz_hm)

        return pred_joints, outs

    # @profile
    def training_step(self, batch, batch_idx):
        b_x, b_y = batch
        del b_y['name']
        for key in b_y.keys():
            raw = b_y[key]
            b_y[key] = raw.reshape(raw.shape[0]*raw.shape[1], *raw.shape[2:])

        # CNN output
        outs = self.model(b_x)
        if self.hparams.use_mask != 'none':
            outs = outs[:3]

        loss = self._calculate_loss3d(outs, b_y)
        self.log('loss', loss.cpu().detach().item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_end(self) -> None:
        return super().on_train_end()

    # @tic_toc
    def validation_step(self, batch, batch_idx):
        loss, results = self._eval(batch, denormalize=False, mode='val')  # Normalized results
        return {'loss':loss, 'results':results}

    # @tic_toc
    def validation_step_end(self, outputs):
        loss = torch.mean(outputs['loss'])
        self.eval_counter += 1
        results = {k:torch.mean(v, axis=0).cpu().detach() for k,v in outputs['results'].items()}
        self.log('val_loss', loss.cpu().detach().item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(results, on_step=False, on_epoch=True, prog_bar=False)
        
    def test_step(self, batch, batch_idx):
        name = batch[1]['name'][0]
        # As here the denormalize is applied, the metrics 
        # will be quite different from val, but they are 
        # all correct. Test version is used for reporting.
        loss, results = self._eval(
            batch, denormalize=True, mode='test'
        )  # Compare denormalized skeletons for test evaluation only

        if self.hparams.ddhp:
            atomized_log_item =  {**results,
                                    'loss': loss.cpu().detach().item(), 
                                    'name': name, 
                                    'subject': name.split('_')[0],
                                    'condition': name.split('_')[1],
                                    'piece': name.split('_')[2]}
        else:
            atomized_log_item = {**{k:v.cpu().detach().numpy() for k,v in results.items()},
                                    'loss': loss.cpu().detach().item(), 
                                    'name': name, 
                                    'subject': name.split('_')[0],
                                    'view': name.split('_')[1],
                                    'lighting': name.split('_')[-1],
                                    'bkg': '_'.join(name.split('_')[2:-1])}
        self.atomized_logs.append(atomized_log_item)
        return {'loss':loss, 'results':results}

    def test_step_end(self, outputs):
        loss = torch.mean(outputs['loss'])
        self.eval_counter += 1
        results = {k:torch.mean(v, axis=0).cpu().detach() for k,v in outputs['results'].items()}
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log_dict(results, on_step=False, on_epoch=True, prog_bar=False)

    def on_test_end(self):
        file_path = op.join(self.hparams.recorder_dir, 
                            f'test_statistics.pkl')
        with open(file_path, 'wb') as f:
            pkl.dump(self.atomized_logs, f)
        self.atomized_logs=[]

    def predict_step(self, b_x, batch_idx=0, mask_input=None):
        # Get prediction
        pred_joints, outs = self(b_x, mask_input=mask_input)
        if self.hparams.use_mask != 'none':
            mask = outs[-1]
            return pred_joints, mask
        else:
            return pred_joints

    def on_epoch_start(self):
        # Make the Progress Bar leave there
        self.eval_counter = 0
        if self.current_epoch > 0:
            print('') #\n
    
    def load_feature_extractor(self):
        feature_extractor = feature_extractor_decorator(
        self.hparams.extractor_name,
        self.hparams.input_channel_num,
        self.hparams.extractor_path,
        self.hparams.extractor_pretrained)
        self.hparams.in_cnn = feature_extractor

    def denormalize_predictions(self, normalized_predictions, b_y):
        """
        Denormalize skeleton prediction and reproject onto original coord system

        Args:
            normalized_predictions (torch.Tensor): normalized predictions
            b_y: batch y object (as returned by 3d joints dataset)

        Returns:
            Returns torch tensor of shape (BATCH_SIZE, NUM_JOINTS, 3)

        Note:
            Prediction skeletons are normalized according to batch depth value
            `z_ref` or torso length

        Todo:
            [] de-normalization is currently CPU only
        """
        device = normalized_predictions.device

        normalized_skeletons = normalized_predictions.cpu()  # CPU only
        pred_skeletons = []
        for i in range(len(normalized_skeletons)):

            denormalization_params = {
                "width": self.hparams.frame_size[1],
                "height": self.hparams.frame_size[0],
                "camera": b_y["camera"][i].cpu(),
            }

            pred_skeleton = Skeleton(normalized_skeletons[i].narrow(-1, 0, 3))
            
            if self.hparams.estimate_depth:
                denormalization_params["torso_length"] = self.hparams.torso_length
            else:
                denormalization_params["z_ref"] = b_y["z_ref"][i].cpu()
            
            # Denormalise the skeleton, adding scale and z position
            pred_skeleton = pred_skeleton.denormalize(
                **denormalization_params
            )._get_tensor()
            
            pred_skeletons.append(pred_skeleton)

        pred_skeletons = torch.stack(pred_skeletons).to(device)
        return pred_skeletons

    def _calculate_loss3d(self, outs, b_y):
        loss = 0
        xy_hms = outs[0]
        zy_hms = outs[1]
        xz_hms = outs[2]

        normalized_skeletons = b_y["normalized_skeleton"]
        b_masks = b_y["mask"]

        for out in zip(xy_hms, zy_hms, xz_hms):
            loss += self.loss_function(out, normalized_skeletons, b_masks)

        return loss / len(outs)

    # @tic_toc
    def _eval(self, batch, denormalize=False, mode='val'):
        """
        Note:
            De-normalization is time-consuming, currently it's performed on CPU
            only. Therefore, it can be specified to either compare normalized or
            de-normalized skeletons
        """
        b_x, b_y = batch
        del b_y['name']

        # Turn the seq_len axis and batch axis into the new batch axis
        for key in b_y.keys():
            raw = b_y[key]
            b_y[key] = raw.reshape(raw.shape[0]*raw.shape[1], *raw.shape[2:])
        
        # Get prediction
        pred_joints, outs = self(b_x)
        if self.hparams.use_mask != 'none':
            mask = outs[-1]
            outs = outs[:3]

        loss = self._calculate_loss3d(outs, b_y)
        
        # denormalize skeletons batch
        if denormalize:
            # The denormalization is only transforming normalized skeletons to SKcam, not SKworld
            pred_joints = self.denormalize_predictions(pred_joints, b_y)
            gt_joints = b_y["skeleton"]  # xyz in original coord        
        else:
            gt_joints = b_y["normalized_skeleton"]  # xyz in normalized coord

        results = {
            # metric_name: metric_function(pred_joints.cpu().detach(), gt_joints.cpu(), b_y["mask"].cpu())#.cpu().detach()
            metric_name: metric_function(pred_joints, gt_joints, b_y["mask"])#.cpu().detach()
            for metric_name, metric_function in self.metrics.items()
        }

        if self.hparams['detailed_test_record'] and mode=='test':
            results['MPJPE_DETAILED'] = self.ori_mpjpe(pred_joints, gt_joints, b_y["mask"])
        
        if self.eval_counter % self.hparams.log_frequency == 0:
            # Record the input x, y, prediction, and mask once in a while 
            b_y_out = {k:v.cpu().detach() for k,v in b_y.items()}
            check_package = {'x':b_x.cpu().detach(), 
                            'y':b_y_out, 
                            'pred_joints':pred_joints.cpu().detach()}
            if self.hparams.use_mask != 'none':
                check_package['mask'] = mask.cpu().detach()

            file_path = op.join(self.hparams.recorder_dir, f'{mode}-batch_{self.eval_counter}.pkl')                  
            with open(file_path, 'wb') as f:
                pkl.dump(check_package, f)
        
        return loss, results

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        # TODO
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        if self.hparams.loss == 'MultiPixelWiseLoss':
            self.loss_function = MultiPixelWiseLoss(self.hparams.reduction, self.hparams.divergence)
        elif self.hparams.loss == 'PixelWiseLoss':
            self.loss_function = PixelWiseLoss(self.hparams.reduction, self.hparams.divergence)

    def configure_metrics(self):
        self.metrics = {}
        if self.hparams.metrics is None:
            self.metrics = {
                "AUC": AUC(reduction=average_loss, auc_reduction=None),
            }
        else:
            if "AUC" in self.hparams.metrics:
                self.metrics["AUC"] = AUC(reduction=average_loss, auc_reduction=None)
            if "MPJPE" in self.hparams.metrics:
                self.metrics["MPJPE"] = MPJPE(reduction=average_loss)
            if "PCK" in self.hparams.metrics:
                self.metrics["PCK"] = PCK(reduction=average_loss)
        
    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except Exception as e:
            print(str(e))
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

def feature_extractor_decorator(model, input_channel_num, extractor_path, extractor_pretrained):
    """ Return a feature extractor based on the model name and other parameters.
    """
    extractor_params = {"n_channels": input_channel_num, "model": model}

    if extractor_path is not None and op.exists(extractor_path):
        extractor_params["custom_model_path"] = extractor_path
    else:
        if extractor_pretrained is not None:
            extractor_params["pretrained"] = extractor_pretrained
        else:
            extractor_params["pretrained"] = True

    feature_extractor = get_feature_extractor(extractor_params)

    return feature_extractor