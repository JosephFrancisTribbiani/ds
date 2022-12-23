import re
import os
import ast
import yaml
import torch
import pandas as pd
from dataclasses import dataclass
from typing import Union, Tuple
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from .dataset import NBMEDataset


def read_yaml(file_path: str = "./nmbe/config.yaml") -> dict:
    """
    Функция для считывания config файла.
    :param file_path: путь к config файлу.
    :return: параметры из config файла.
    """
  
    with open(file=file_path, mode='r') as f:
        try:
            loader = yaml.SafeLoader
    
            # добавляем возможность считывать числа, записанные в YAML файл в формате 1e-4, к примеру
            loader.add_implicit_resolver(
                u'tag:yaml.org,2002:float',
                re.compile(u'''^(?:
                               [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                               |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                               |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                               |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                               |[-+]?\\.(?:inf|Inf|INF)
                               |\\.(?:nan|NaN|NAN))$''', re.X),
                list(u'-+0123456789.')
            )
    
            return yaml.load(f, Loader=loader)
        except yaml.YAMLError as exc:
            print(exc)


@dataclass
class ModelConfig:
    checkpoint: str = "microsoft/deberta-base"
    fc_dropout: Union[int, float] = 0.2
    use_pretrained: bool = True
    config_path: str = None
    output_hidden_states: bool = True
    num_labels: int = 1


@dataclass
class DataConfig:
    max_length: int = 512
    batch_size: int = 12
    n_folds: int = 5
    n_workers: int = 4


@dataclass
class TrainConfig:
    n_epochs: int = 5
    device: str = "cuda"
    backbone_lr: Union[int, float] = 2e-5
    fc_lr: Union[int, float] = 2e-5
    weight_decay: Union[int, float] = 0.01
    betas: Tuple[Union[int, float], Union[int, float]] = (0.9, 0.999)
    eps: Union[int, float] = 1e-8
    scheduler_type: str = "linear"
    num_warmup_steps: int = 0
    num_cycles: Union[int, float] = 0.5
    use_autocast: bool = True
    iters_to_accumulate: int = 6
    max_gradient_norm: int = 4
    verbose_step: int = None


def prepare_dataloaders(dataset: pd.DataFrame, tokenizer: torch.nn, fold: int, max_length: int, 
                        batch_size: int, n_workers: int) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare train and eval dataloaders for specified fold for CV.
    :param dataset: dataframe with column "fold", which specified the fold number
    :param tokenzier: autotokenizer from hugging face
    :param fold: number of fold using for eval set
    :param batch_size: batch size - the parameter of a DataLoader
    :n_workers: number of workers - the parameter of a DataLoader
    :return: DataLoader for training, size of train set and DataLoader for evaluation
    """
    # trainloader preparation
    train_df = dataset[dataset['fold'] != fold]
    train_df = train_df.reset_index(drop=True, inplace=False)
    trainset = NBMEDataset(tokenizer=tokenizer, feature_texts=train_df['feature_text'].values, 
                           pn_histories=train_df['pn_history'].values, locations=train_df['location'].values,
                           max_length=max_length)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, 
                             num_workers=n_workers, pin_memory=True, drop_last=True)
    
    # evalloader preparation
    eval_df = dataset[dataset['fold'] == fold]
    eval_df = eval_df.reset_index(drop=True, inplace=False)
    evalset = NBMEDataset(tokenizer=tokenizer, feature_texts=eval_df['feature_text'].values, 
                          pn_histories=eval_df['pn_history'].values, locations=eval_df['location'].values,
                          max_length=max_length)
    evalloader = DataLoader(dataset=evalset, batch_size=batch_size, shuffle=False, 
                            num_workers=n_workers, pin_memory=True, drop_last=False)
    return trainloader, evalloader


def cv_split(data: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """
    Split data into folds using GroupKFold from sklearn, column "pn_num" for groupping. 
    Add column numeber to specify a number of a fold.
  
    :param data: input dataframe with columns "pn_num" and "loaction"
    :type data: pd.DataFrame.
    :param n_folds: the number of folds to split the data
    :type n_folds: int
    :rtype: pd.DataFrame
    :return: input data with new column "fold" specifies a number of a fold
    """
    assert {"pn_num", "location"}.issubset(set(data.columns)), 'The data must necessarily contain columns "location" and "pn_num"'
    assert n_folds > 0, "Number of folds must be at least 1"
  
    if n_folds == 1:
        data['fold'] = 0
    else:
        splitter = GroupKFold(n_splits=n_folds)
        groups = data["pn_num"].values
        for n, (_, val_indexes) in enumerate(splitter.split(data, data["location"], groups)):
            data.loc[val_indexes, "fold"] = n
        data["fold"] = data["fold"].astype("int8")
    return data


def read_input_data(loc: str = "./input_data/") -> pd.DataFrame:
    """
    Function for the input data loading and merging.
    :param loc: root location with input data, defaults to ./input_data/
    """
    # loading the data
    # train
    train = pd.read_csv(os.path.join(loc, "train.csv"))
    train["annotation"] = train["annotation"].apply(ast.literal_eval)
    train["location"] = train["location"].apply(ast.literal_eval)

    # features
    features = pd.read_csv(os.path.join(loc, "features.csv"))

    # patient_notes
    patient_notes = pd.read_csv(os.path.join(loc, "patient_notes.csv"))

    # merge the input data
    train_joined = train.merge(features, how='left', on=['feature_num', 'case_num'])
    train_joined = train_joined.merge(patient_notes, how='left', on=['pn_num', 'case_num'])
    return train_joined
