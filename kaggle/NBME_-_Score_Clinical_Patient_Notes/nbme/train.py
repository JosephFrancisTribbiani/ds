import gc
import os
import logging
import numpy as np
from typing import List
from tqdm.auto import tqdm

from nbme.model import NBMEModel
from nbme.utils import ModelConfig, TrainConfig

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s - %(message)s',
)
LOGGER = logging.getLogger(__name__)


class TrainModel:

    def __init__(self, model_cfg: ModelConfig, num_training_steps: int, train_cfg: TrainConfig, 
                 writer_path: str = "./runs/", suffix: str = None) -> None:
        # training loop configuration
        self._train_cfg = train_cfg

        # set device
        self._device = self._train_cfg.device if torch.cuda.is_available() else "cpu"

        # set model
        self._model = self._model_initialization(config=model_cfg)
        self._model.to(self._device)

        # configure scaler, optimizer and scheduler
        self._scaler = torch.cuda.amp.GradScaler(enabled=(self._device == "cuda"))
        self._optimizer, self._scheduler = self._set_configurator(num_training_steps=num_training_steps)

        # set loss function
        # reduction means do not to apply aggregation function (as sum or mean)
        # in reason to masking loss (do not penalize special tokens predict - [CLS], [PAD] and etc.)
        self._loss_func = nn.BCEWithLogitsLoss(reduction="none")
        self._best_loss_eval = float("inf")

        # set writer
        if not os.path.exists(writer_path):
            os.makedirs(writer_path)
        self._writer = SummaryWriter(writer_path, filename_suffix=suffix)
        self._global_step_train = 0
        self._global_step_eval = 0

    @staticmethod
    def _model_initialization(config: ModelConfig) -> nn.Module:
        LOGGER.info("Model initialization")
        model = NBMEModel(**config.__dict__)
        LOGGER.info("Done")
        return model
  
    def _get_trainable_params(self) -> List[dict]:
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        trainable_params = [
            {'params': [param for node, param in self._model.backbone.named_parameters() if not node in no_decay], 
             'lr': self._train_cfg.backbone_lr, 'weight_decay': self._train_cfg.weight_decay}, 
            {'params': [param for node, param in self._model.backbone.named_parameters() if node in no_decay],
             'lr': self._train_cfg.backbone_lr, 'weight_decay': 0.0},
            {'params': [param for _, param in self._model.fc.named_parameters()],
             'lr': self._train_cfg.fc_lr, 'weight_decay': 0.0}]
        return trainable_params

    def _set_configurator(self, num_training_steps: int):
        # configure optimizer
        trainable_params = self._get_trainable_params()
        optimizer = AdamW(
            params=trainable_params, lr=self._train_cfg.backbone_lr, betas=self._train_cfg.betas,
            eps=self._train_cfg.eps)
    
        # configure scheduler
        scheduler = None
        if self._train_cfg.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self._train_cfg.num_warmup_steps,
                num_training_steps=num_training_steps)
        elif self._train_cfg.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=self._train_cfg.num_warmup_steps,
                num_training_steps=num_training_steps, num_cycles=self._train_cfg.num_cycles)
        return optimizer, scheduler

    def fit(self, trainloader: DataLoader, evalloader: DataLoader, model_loc: str = None) -> None:
        # reset global step
        self._global_step = 0
        self._global_step_eval = 0
    
        # iterate over epochs
        LOGGER.info("Iterate over epochs")
        for epoch in tqdm(range(self._train_cfg.n_epochs)):
            LOGGER.info("Epoch [{}]/[{}]".format(epoch + 1, self._train_cfg.n_epochs))
            avg_loss_train = self.train_nn(trainloader=trainloader)
      
            with torch.no_grad():
                avg_loss_eval = self.eval_nn(evalloader=evalloader)
      
            # plot metrics using tensorboard
            LOGGER.info("Train average loss: {:.4f}".format(avg_loss_train))
            LOGGER.info("Eval average loss: {:.4f}".format(avg_loss_eval))
            self._writer.add_scalar('train/loss_avg', avg_loss_train, epoch + 1)
            self._writer.add_scalar('eval/loss_avg', avg_loss_eval, epoch + 1)

            if model_loc and avg_loss_eval < self._best_loss_eval:
                torch.save({"epoch": epoch + 1, "model_state_dict": self._model.state_dict(), 
                           "loss": avg_loss_eval}, model_loc)
                LOGGER.info("Model saved")  
            self.best_loss_eval = min(avg_loss_eval, self.best_loss_eval)
              
            # clean up cache
            if self._device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
        return

    def train_nn(self, trainloader: DataLoader) -> list:
        LOGGER.info("Train loop")
        running_loss = list()
    
        self._model.train()
        for step, (inputs, labels) in enumerate(trainloader):
            # reset peaks of CUDA usage
            if self._device == "cuda":
                torch.cuda.reset_peak_memory_stats(device=self._device)
      
            # relocate inputs and labels to device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            labels = labels.to(self._device)
      
            # using autocasting to cast gradients from float32 to float16
            # forward pass
            with torch.autocast(device_type=self._device, enabled=((self._train_cfg.use_autocast) & (self._device == "cuda"))):
                outputs = self._model(inputs)
                loss = self._loss_func(outputs.view(-1, 1), labels.view(-1, 1))
                loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
                running_loss.append(loss.cpu())
                # torch.masked_select gets tensor and boolean mask (like [True, False, ..., False]) as input
                # and returns 1D tensor with values which has True value in corresponding element in mask
                loss = loss / self._train_cfg.iters_to_accumulate
                # using gradients accumulation due to small batch size

            # backward pass with gradient scaling
            self._scaler.scale(loss).backward()

            # update model's weights and update scaling rate
            if ((step + 1) % self._train_cfg.iters_to_accumulate == 0) or ((step + 1) == len(trainloader)):
                # before gradient clipping firstly we have to unscale the gradients, otherwise we will clip scaled gradients
                # pytorch documentation: https://pytorch.org/docs/stable/notes/amp_examples.html
                self._scaler.unscale_(self._optimizer)

                # clip gradients to avoid blow up of the calculated gradients
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=self._train_cfg.max_gradient_norm)

                # update weights and learning rate
                self._scaler.step(self._optimizer)
                self._scaler.update()
                if self._scheduler:
                    self._scheduler.step()

                # set gradients to zero
                self._optimizer.zero_grad()

            # plot metrics results
            if self._train_cfg.verbose_step and not ((step + 1) % self._train_cfg.verbose_step):
                # print results
                LOGGER.info("Step [{}] loss: {:.4f}".format(step + 1, running_loss[-1]))

                # using tensorboard
                self._writer.add_scalars(main_tag="learning rates", tag_scalar_dict=dict(zip(["bert", "bert_bias", "classifier"], 
                                         self._scheduler.get_last_lr())), global_step=self._global_step_train + 1)
                self._writer.add_scalar('train/loss', running_loss[-1], self._global_step_train + 1)
                if self._device == "cuda":
                    mm_allocated = torch.cuda.max_memory_allocated(device=self._device)
                    mm_reserved = torch.cuda.max_memory_reserved(device=self._device)
                    self._writer.add_scalar(
                        'cuda/max_memory_allocated', mm_allocated, self._global_step_train + self._global_step_eval + 1)
                    self._writer.add_scalar(
                        'cuda/max_related_memory_allocated', round(100 * mm_allocated / mm_reserved, 2), 
                        self._global_step_train + self._global_step_eval + 1)
                    
            # increase global step value
            self._global_step_train += 1
        return np.mean(running_loss)

    def eval_nn(self, evalloader: DataLoader) -> list:
        LOGGER.info("Eval loop")
        running_loss = list()
    
        self._model.eval()
        for step, (inputs, labels) in enumerate(evalloader):
            # reset peaks of CUDA usage
            if self._device == "cuda":
                torch.cuda.reset_peak_memory_stats(device=self._device)
      
            # relocate inputs and labels to device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            labels = labels.to(self._device)
      
            # forward pass
            outputs = self._model(inputs)
            loss = self._loss_func(outputs.view(-1, 1), labels.view(-1, 1))
            loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
            running_loss.append(loss.cpu())
      
            # plot metrics results
            if self._train_cfg.verbose_step and not ((step + 1) % self._train_cfg.verbose_step):
                # print results
                LOGGER.info("Step [{}] loss:\t{:.4f}".format(step + 1, running_loss[-1]))

                self._writer.add_scalar('eval/loss', running_loss[-1], self._global_step_eval + 1)
                if self._device == "cuda":
                    mm_allocated = torch.cuda.max_memory_allocated(device=self._device)
                    mm_reserved = torch.cuda.max_memory_reserved(device=self._device)
                    self._writer.add_scalar(
                        'cuda/max_memory_allocated', mm_allocated, self._global_step_train + self._global_step_eval + 1)
                    self._writer.add_scalar(
                        'cuda/max_related_memory_allocated', round(100 * mm_allocated / mm_reserved, 2), 
                        self._global_step_train + self._global_step_eval + 1)
                    
            # increase global step value
            self._global_step_eval += 1
            