# Copyright 2024 Haowen Yu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import torch
import importlib
import torch.optim.lr_scheduler as lrs
import lightning.pytorch as pl

from utils.metrics.classification import top1_accuracy, top5_accuracy
from typing import Callable, Dict, Tuple

from loss.contrastive_learning import global_multipos_info_nce, dense_info_nce
from loss.reconstruction import recon_loss


class ContrastiveLearningModelInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.__load_model()
        self.loss_function = self.__configure_loss()

        # Parse model type
        self.model_class_name = str(self.hparams.model_class_name)
        if self.model_class_name != 'unimodal':
            self.global_CL_lambda = float(self.hparams.global_CL_lambda)
            self.global_multipos_info_nce_temperature = float(self.hparams.global_multipos_info_nce_temperature)
            self.global_multipos_info_nce_eps = float(self.hparams.global_multipos_info_nce_eps)
            self.dense_CL_lambda = float(self.hparams.dense_CL_lambda)
            self.dense_info_nce_temperature = float(self.dense_info_nce_temperature)
            self.dense_info_nce_include_spatial_negatives = bool(self.hparams.dense_info_nce_include_spatial_negatives)
            self.dense_info_nce_neighborhood = int(self.hparams.dense_info_nce_neighborhood)
            self.dense_info_nce_eps = float(self.hparams.dense_info_nce_eps)
            print(f'raw global_multipos_info_nce_temperature: {self.hparams.global_multipos_info_nce_temperature} type: {type(self.hparams.global_multipos_info_nce_temperature)} parsed: {self.hparams.global_multipos_info_nce_temperature}')
            print(f'raw global_multipos_info_nce_temperature: {self.hparams.global_multipos_info_nce_eps} type: {type(self.hparams.global_multipos_info_nce_eps)} parsed: {self.hparams.global_multipos_info_nce_eps}')
            print(f'raw global_multipos_info_nce_temperature: {self.hparams.dense_info_nce_temperature} type: {type(self.hparams.dense_info_nce_temperature)} parsed: {self.hparams.dense_info_nce_temperature}')
            print(f'raw global_multipos_info_nce_temperature: {self.hparams.dense_info_nce_include_spatial_negatives} type: {type(self.hparams.dense_info_nce_include_spatial_negatives)} parsed: {self.hparams.dense_info_nce_include_spatial_negatives}')
            print(f'raw global_multipos_info_nce_temperature: {self.hparams.dense_info_nce_neighborhood} type: {type(self.hparams.dense_info_nce_neighborhood)} parsed: {self.hparams.dense_info_nce_neighborhood}')
            print(f'raw global_multipos_info_nce_temperature: {self.hparams.dense_info_nce_eps} type: {type(self.hparams.dense_info_nce_eps)} parsed: {self.hparams.dense_info_nce_eps}')

    def forward(self, x):
        return self.model(x)
    
    # For all these hook functions like on_XXX_<epoch|batch>_<end|start>(), 
    # check document: https://lightning.ai/docs/pytorch/LTS/common/lightning_module.html
    # Epoch level training logging
    def on_train_epoch_end(self):
        pass

    # Epoch level validation logging
    def on_validation_epoch_end(self):
        pass

    # Epoch level testing logging
    def on_test_epoch_end(self):
        pass

    # Caution: self.model.train() is invoked
    def training_step(self, batch, batch_idx):
        train_input, train_labels = batch
        train_out = self(train_input)
        train_loss_dict = self.loss_function(train_out, train_labels, 'val')
        loss_train = train_loss_dict['loss']

        # Metrics
        pos_sim_mean = train_loss_dict['pos_sim_mean']
        neg_sim_mean = train_loss_dict['neg_sim_mean']
        self.log('train_pos_sim_mean', pos_sim_mean.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_neg_sim_mean', neg_sim_mean.item(), on_step=True, on_epoch=True, prog_bar=True)

        return {
            'loss': loss_train
        }

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def validation_step(self, batch, batch_idx):
        if self.model_class_name == 'unimodal':
            # TODO: add specific transformations for each representation, if necessary
            val_input = batch
            val_out = self(val_input['repr_data'][0])
            val_recon_loss = self.loss_function(val_out, val_input)
            self.log('val_recon_loss', val_recon_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
            # TODO: add PSNR & LPIPS metrics
            # TODO: which embedding to use for the full autoencoder?
            
            
        elif self.model_class_name == 'pairwise':
            # TODO: add specific transformations for each representation, if necessary
            val_input, val_labels = batch
            val_out_dict = self(val_input)
            repr1_feature_map = val_out_dict['repr1_feature_map']
            repr2_feature_map = val_out_dict['repr2_feature_map']
            stacked_feature_map = torch.concat([repr1_feature_map.unsqueeze(1), repr2_feature_map.unsqueeze(1)], dim=1)
            repr1_embedding = val_out_dict['repr1_embedding']
            repr2_embedding = val_out_dict['repr2_embedding']
            stacked_embedding = torch.concat([repr1_embedding.unsqueeze(1), repr2_embedding.unsqueeze(1)], dim=1)
            val_loss_dict = self.loss_function(stacked_embedding, stacked_feature_map)

            loss_val = val_loss_dict['total_loss']
            self.log('val_CL_loss', loss_val.item(), on_step=True, on_epoch=True, prog_bar=True)
            self.log('val_global_CL_loss', val_loss_dict['global_CL_loss'], on_step=True, on_epoch=True, prog_bar=True)
            # self.log('val_dense_CL_loss', val_loss_dict['dense_CL_loss'], on_step=True, on_epoch=True, prog_bar=True)

            # Metrics
            global_pos_sim_mean = val_loss_dict['global_pos_sim_mean']
            global_neg_sim_mean = val_loss_dict['global_neg_sim_mean']
            # dense_pos_sim_mean = val_loss_dict['dense_pos_sim_mean']
            # dense_neg_sim_mean = val_loss_dict['dense_neg_sim_mean']
            self.log('val_global_pos_sim_mean', global_pos_sim_mean.item(), on_step=True, on_epoch=True, prog_bar=True)
            self.log('val_global_neg_sim_mean', global_neg_sim_mean.item(), on_step=True, on_epoch=True, prog_bar=True)
            # self.log('val_dense_pos_sim_mean', dense_pos_sim_mean.item(), on_step=True, on_epoch=True, prog_bar=True)
            # self.log('val_dense_neg_sim_mean', dense_neg_sim_mean.item(), on_step=True, on_epoch=True, prog_bar=True)

            return {
                'loss': loss_val
            }

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def test_step(self, batch, batch_idx):
        test_input, test_labels = batch
        test_out = self(test_input)
        test_loss = self.loss_function(test_out, test_labels, 'test')

        test_step_output = {
            'loss': test_loss,
            'pred': test_out,
            'ground_truth': test_labels
        }

        self.test_epoch_output.append(test_step_output)

        return test_step_output

    # When there are multiple optimizers, modify this function to fit in your needs
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay)
        )

        # No learning rate scheduler, just return the optimizer
        if self.hparams.lr_scheduler is None:
            return [optimizer]

        # Return tuple of optimizer and learning rate scheduler
        if self.hparams.lr_scheduler == 'step':
            scheduler = lrs.StepLR(
                optimizer,
                step_size=self.hparams.lr_decay_epochs,
                gamma=self.hparams.lr_decay_rate
            )
        elif self.hparams.lr_scheduler == 'cosine':
            scheduler = lrs.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.lr_decay_epochs,
                eta_min=self.hparams.lr_decay_min_lr
            )
        else:
            raise ValueError('Invalid lr_scheduler type!')
        return [optimizer], [scheduler]

    def __configure_loss(self):
        def unimodal_loss_func(preds, labels):
            return recon_loss(pred=preds, gt=labels)

        def contrastive_loss_func(embeddings, feature_maps):
            global_CL_loss_dict = global_multipos_info_nce(
                z=embeddings, 
                temperature=self.global_multipos_info_nce_temperature, 
                eps=self.global_multipos_info_nce_eps
            )
            """dense_CL_loss_dict = dense_info_nce(
                maps=feature_maps, 
                temperature=self.dense_info_nce_temperature, 
                include_spatial_negatives=self.dense_info_nce_include_spatial_negatives, 
                neighborhood=self.dense_info_nce_neighborhood, 
                eps=self.dense_info_nce_eps
            )"""

            total_loss = self.global_CL_lambda * global_CL_loss_dict['loss'] # + self.dense_CL_lambda * dense_CL_loss_dict['loss']

            return {
                'global_CL_loss': global_CL_loss_dict['loss'],
                'global_pos_sim_mean': global_CL_loss_dict['pos_sim_mean'],
                'global_neg_sim_mean': global_CL_loss_dict['neg_sim_mean'],
                """'dense_CL_loss': dense_CL_loss_dict['loss'],
                'dense_pos_sim_mean': dense_CL_loss_dict['pos_sim_mean'],
                'dense_neg_sim_mean': dense_CL_loss_dict['neg_sim_mean'],"""
                'total_loss': total_loss
            }


        if self.hparams.model_class_name == 'unimodal':
            return unimodal_loss_func
        else:
            return contrastive_loss_func

    def __load_model(self):
        name = self.hparams.model_class_name
        # Attempt to import the `CamelCase` class name from the `snake_case.py` module. The module should be placed
        # within the same folder as model_interface.py. Always name your model file name as `snake_case.py` and
        # model class name as corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            model_class = getattr(importlib.import_module('model.' + name, package=__package__), camel_name)
        except Exception:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        model = self.__instantiate(model_class)
        if self.hparams.use_compile:
            torch.compile(model)
        return model

    def __instantiate(self, model_class, **other_args):
        # Instantiate a model using the imported class name and parameters from self.hparams dictionary.
        # You can also input any args to overwrite the corresponding value in self.hparams.
        target_args = inspect.getfullargspec(model_class.__init__).args[1:]
        this_args = self.hparams.keys()
        merged_args = {}
        # Only assign arguments that are required in the user-defined torch.nn.Module subclass by their name.
        # You need to define the required arguments in main function.
        for arg in target_args:
            if arg in this_args:
                merged_args[arg] = getattr(self.hparams, arg)

        merged_args.update(other_args)
        return model_class(**merged_args)
