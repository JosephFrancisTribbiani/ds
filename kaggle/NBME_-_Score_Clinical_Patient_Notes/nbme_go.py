import os
import pprint
import logging
import argparse
from transformers import AutoTokenizer
from nbme.train import TrainModel
from nbme.utils import ModelConfig, DataConfig, TrainConfig, read_yaml
from nbme.utils import read_input_data, prepare_dataloaders, cv_split


logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s - %(message)s',
)
LOGGER = logging.getLogger(__name__)


def main(args) -> None:
    print(args)
    # loading configuration data
    LOGGER.info("Loading the configuration files")
    config_file = read_yaml(file_path=args.config)
    model_cfg =     ModelConfig(**config_file.get("model"))
    data_cfg =      DataConfig( **config_file.get("data"))
    train_cfg =     TrainConfig(**config_file.get("train"))
    pp = pprint.PrettyPrinter(indent=4)
    LOGGER.info("Model configuration:")
    pp.pprint(model_cfg.__dict__)
    LOGGER.info("Input data configuration:")
    pp.pprint(data_cfg.__dict__)
    LOGGER.info("Training loop configuration:")
    pp.pprint(train_cfg.__dict__) 

    # prepare the input data
    LOGGER.info("Loading the input data")
    dataset = read_input_data(loc=args.data)
    LOGGER.info("Input data shape: {}".format(dataset.shape))

    # loading pretrained tokenizer
    LOGGER.info("Loading pretrained tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.checkpoint)
    LOGGER.info("Done")

    # split data into folds
    LOGGER.info("Splitting data into %d fold(s)...", data_cfg.n_folds)
    dataset = cv_split(data=dataset, n_folds=data_cfg.n_folds)
    LOGGER.info("Done")

    # iterating over folds
    LOGGER.info("Iterating over folds")
    for curr_fold in range(data_cfg.n_folds):
        LOGGER.info("Fold: {}/{}".format(curr_fold + 1, data_cfg.n_folds))

        # prepare dataloader
        LOGGER.info("Train / Eval dataloaders preparation")
        trainloader, evalloader = prepare_dataloaders(
            dataset=dataset, tokenizer=tokenizer, fold=curr_fold, max_length=data_cfg.max_length,
            batch_size=data_cfg.batch_size, n_workers=data_cfg.n_workers)
        num_training_steps = len(trainloader) * train_cfg.n_epochs
        LOGGER.info("Done")

        # train model from the current fold
        if args.save and not os.path.exists(args.save):
            os.makedirs(args.save)
        tm = TrainModel(model_cfg=model_cfg, num_training_steps=num_training_steps, train_cfg=train_cfg, 
                        writer_path=args.run, suffix=f"fold_{curr_fold + 1}")
        tm.fit(trainloader=trainloader, evalloader=evalloader, 
               model_loc=os.path.join(args.save, f"model_fold_{curr_fold + 1}.pth"))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for NER of patient notes")
    parser.add_argument("--data", type=str, default="./input_data/", 
                        help="Location of the input data.")
    parser.add_argument("--config", type=str, default="./nbme/config.yaml", 
                        help="Location of the configuration file.")                       
    parser.add_argument("--save", type=str, default="./weights/", 
                        help="Location to save trained models.")
    parser.add_argument("--run", type=str, default="./runs/", 
                        help="Location to save trainig metrics.")
    args = parser.parse_args()
    main(args=args)
