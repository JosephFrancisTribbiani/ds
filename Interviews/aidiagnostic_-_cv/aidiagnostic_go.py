import pprint
import logging
import argparse
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, random_split
from aidiagnostic.utils import read_yaml, ModelConfig, DataSetConfig, DataLoaderConfig
from aidiagnostic.dataset import AiDiagnosticDataset
from aidiagnostic.train import TrainSegmentationModel


logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
)
LOGGER = logging.getLogger(__name__)


def main(args) -> None:
  # read configuration
  LOGGER.info("Read the configuration file from {}".format(args.config_loc))
  config_file = read_yaml(file_path=args.config_loc)
  model_cfg =      ModelConfig(     **config_file.get("model"))
  dataset_cfg =    DataSetConfig(   **config_file.get("dataset"))
  dataloader_cfg = DataLoaderConfig(**config_file.get("dataloader"))
  pp = pprint.PrettyPrinter(indent=4)
  LOGGER.info("Model configuration:")
  pp.pprint(model_cfg.__dict__)
  LOGGER.info("Dataset configuration:")
  pp.pprint(dataset_cfg.__dict__)
  LOGGER.info("Dataloader configuration:")
  pp.pprint(dataloader_cfg.__dict__)   

  # prepare train and eval data
  # initialize dataset
  LOGGER.info("Dataset initialization")
  dataset = AiDiagnosticDataset(**dataset_cfg.__dict__)

  # train eval split
  LOGGER.info("Train / Eval split")
  trainsize = int(dataset.__len__() * args.train_size)
  evalsize = dataset.__len__() - trainsize
  trainset, evalset = random_split(dataset, [trainsize, evalsize])
  LOGGER.info("\tTrain size:\t{}".format(trainsize))
  LOGGER.info("\tEval size:\t{}".format(evalsize))

  # initialize dataloaders
  LOGGER.info("DataLoaders initialization")
  trainloader = DataLoader(trainset, **dataloader_cfg.__dict__, shuffle=True)
  evalloader = DataLoader(evalset, **dataloader_cfg.__dict__, shuffle=False)
  LOGGER.info("\tTrain DataLoader size:\t{}".format(len(trainloader)))
  LOGGER.info("\tEval DataLoader size:\t{}".format(len(evalloader)))

  # model initialization
  LOGGER.info("Load the model from segmentation_models_pytorch")
  model = smp.UnetPlusPlus(**model_cfg.__dict__)

  LOGGER.info("Train the loaded model")
  tm = TrainSegmentationModel(model=model, device=args.device, use_dice=args.dice, use_bce=args.bce, lr=args.lr, 
                              patience=args.patience, factor=args.factor, writer_path=args.writer_path, model_loc=args.model_loc)
  tm.fit(trainloader=trainloader, evalloader=evalloader, n_epochs=args.epochs)
  LOGGER.info("Finished successfully")
  return


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Training loop arguments parser for lungs segmentation.")
  parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for model training")
  parser.add_argument("--dice", action="store_true", help="Specify if Dice loss to use during the training process")
  parser.add_argument("--bce", action="store_true", help="Specify if BCE loss to use during the training process")
  parser.add_argument("--lr", type=float, default=3e-3, help="The learning rate for the training process")
  parser.add_argument("--patience", type=int, default=10, help="Number of epochs with no improvement after which learning " \
                                                               "rate will be reduced. For example, if patience = 2, then we " \
                                                               "will ignore the first 2 epochs with no improvement, and will " \
                                                               "only decrease the LR after the 3rd epoch if the loss still " \
                                                               "hasn’t improved then. Default: 10")
  parser.add_argument("--factor", type=float, default=0.1, help="Factor by which the learning rate will be reduced. " \
                                                                "new_lr = lr * factor. Default: 0.1")
  parser.add_argument("--writer-path", type=str, default="./runs/", help="Save directory location for torch tensorboard")
  parser.add_argument("--model-loc", default=None, help="Save directory location for trained model")
  parser.add_argument("--config-loc", type=str, default="./config.yaml", help="Location of the config file")
  parser.add_argument("-ts", "--train-size", type=float, default=0.8, help="Size of training dataset")
  parser.add_argument("-e", "--epochs", type=int, default=40, help="Number of training epochs")

  args = parser.parse_args()

  main(args)
