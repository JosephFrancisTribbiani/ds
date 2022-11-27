import gc
import logging
from typing import Union, Tuple
from pathlib import Path
from tqdm.auto import tqdm
from .utils import RunningLoss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from segmentation_models_pytorch.losses import DiceLoss

logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
)
LOGGER = logging.getLogger(__name__)


class TrainSegmentationModel:

  def __init__(self, model: torch.nn, device: str = "cuda", use_dice: bool = True, use_bce: bool = True, 
               lr: Union[int, float] = 3e-3, patience: int = 10, factor: float = 0.05, writer_path: str = "./runs/", 
               model_loc: str = None):

    self.device = device if torch.cuda.is_available() else "cpu"
    self.model = model.to(self.device)
    self.model_loc = model_loc
    self.best_loss_eval = float("inf")

    # define loss functions
    self.criterions = dict()
    if use_dice:
      self.criterions["dice"] = DiceLoss(mode="binary", from_logits=True)
    if use_bce:
      self.criterions["bce"] = nn.BCEWithLogitsLoss(reduction="mean")

    # set optimizer and scheduler
    self.optimizer, self.scheduler = \
      self.set_optim(trainable_params=self.model.parameters(), lr=lr, patience=patience, factor=factor)
    
    # set writer
    self.writer = None
    if writer_path is not None:
      self.writer = SummaryWriter(writer_path)

  @staticmethod
  def set_optim(trainable_params, lr: Union[int, float] = 3e-3, patience: int = 10, factor: float = 0.05):
    # configure optimizer
    optimizer = optim.AdamW(params=trainable_params, lr=lr)
    
    # configure scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", patience=patience, factor=factor)
    return optimizer, scheduler

  @staticmethod
  def calculate_loss(y_pred: torch.tensor, y_true: torch.tensor, criterions: dict) -> Tuple[dict, torch.tensor]:
    losses = {k: lossf(y_pred, y_true) for k, lossf in criterions.items()}
    loss_total = 0
    for _, loss in losses.items():
      loss_total += loss
    return losses, loss_total
  
  def plot_metrics(self, rl: dict, global_step: int, root_tag: str = "train") -> None:
    for k, v in rl.items():
      LOGGER.info("\t\t{}:\t{:.4f}".format(k, v))
      if self.writer is not None:
        self.writer.add_scalar(tag=str(Path(root_tag) / k), scalar_value=v, global_step=global_step)
    return

  def fit(self, trainloader: DataLoader, evalloader: DataLoader, n_epochs: int = 64) -> None:
    LOGGER.info("Iterate over epochs")
    for epoch in range(n_epochs):
      LOGGER.info("\tEpoch [{}]/[{}]".format(epoch, n_epochs))

      curr_lr = self.scheduler.optimizer.param_groups[0]['lr']
      LOGGER.info("\t\tLR:\t{}".format(curr_lr))
      if self.writer is not None:
        self.writer.add_scalar(tag="lr", scalar_value=curr_lr, global_step=epoch)

      LOGGER.info("\t\tTrain loop")
      rl_train = self.train_nn(trainloader=trainloader)
      self.plot_metrics(rl=rl_train.running_loss, global_step=epoch, root_tag="train")

      with torch.no_grad():
        LOGGER.info("\t\tEval loop")
        rl_eval = self.eval_nn(evalloader=evalloader)
        self.plot_metrics(rl=rl_eval.running_loss, global_step=epoch, root_tag="eval")

      # scheduler step
      cur_eval_loss = rl_eval.get_sum()
      self.scheduler.step(cur_eval_loss)

      if self.model_loc is not None and cur_eval_loss < self.best_loss_eval:
        torch.save({'epoch': epoch, 
                    'model_state_dict': self.model.state_dict(),
                    'loss': cur_eval_loss}, self.model_loc)
        LOGGER.info("\t\tModel saved")  
      self.best_loss_eval = min(cur_eval_loss, self.best_loss_eval)

      # clean up cache
      if self.device == "cuda":
        LOGGER.info("\t\tClean up CUDA memory") 
        torch.cuda.empty_cache()
        gc.collect()
    return

  def train_nn(self, trainloader: DataLoader):
    self.model.train()
    rl = RunningLoss()

    for n_iter, (input, labels) in tqdm(enumerate(trainloader), total=len(trainloader)):
      input = input.to(self.device)
      labels = labels.to(self.device)
      self.optimizer.zero_grad()

      # forward pass
      output = self.model(input)

      # loss calculation
      losses, loss_total = self.calculate_loss(y_pred=output, y_true=labels, criterions=self.criterions)

      # backward and weight updation
      loss_total.backward()
      self.optimizer.step()
      rl.update(iter=n_iter, **losses)
    return rl

  def eval_nn(self, evalloader: DataLoader):
    self.model.eval()
    rl = RunningLoss()

    for n_iter, (input, labels) in tqdm(enumerate(evalloader), total=len(evalloader)):
      input = input.to(self.device)
      labels = labels.to(self.device)

      # forward pass
      output = self.model(input)

      # loss calculation
      losses, _ = self.calculate_loss(y_pred=output, y_true=labels, criterions=self.criterions)
      rl.update(iter=n_iter, **losses)
    return rl
    