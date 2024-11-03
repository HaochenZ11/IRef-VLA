'''
Parts adapted from: MVT-3DVG repositories
https://github.com/sega-hsj/MVT-3DVG
'''

import numpy as np
import os
import multiprocessing as mp
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import argparse
import csv

def get_nyu_classes(csv_path, pad_idx):
    '''
    Gets mapping of class ids to class names for all NYUv2 classes
    :return: Dict of class_id:class_name
    '''
    idx_to_class = {}
    with open(csv_path, encoding='utf-8') as csv_file:
        csvReader = csv.DictReader(csv_file)
        for row in csvReader:
            idx_to_class.update({int(row["nyuId"]):row["nyuClass"]})

    idx_to_class.update({pad_idx:'PAD'})
    return idx_to_class

def get_num_classes(train_loader, test_loader, pad_idx):
    '''
    Gets total number of unique classes in dataset
    :return: List of all unique class ids
    '''
    class_list = []
    if train_loader is not None:
        for batch in train_loader:
            classes = batch["class_labels"].flatten()
            class_list += classes.tolist()
    if test_loader is not None:
        for batch in test_loader:
            classes = batch["class_labels"].flatten()
            class_list += classes.tolist()

    all_classes = set(class_list)
    if pad_idx in all_classes:
        all_classes.remove(pad_idx)
        
    return all_classes


def str2bool(v):
    '''
    Converts string arg input into boolean value
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_test_split(train_size, dataset):
    '''
    Randomly splits dataset based on train_size and returns the splits
    '''
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    print("Splitting dataset of len: ", len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def dataset_to_dataloader(dataset, args, train=True):
    '''
    Converts dataset into dataloader
    :return: DataLoader object
    '''
    #g = torch.Generator()
    #g.manual_seed(0)
    if len(dataset) == 0:
        return None

    if train: # if eval all data
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
            #generator=g
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            #generator=g
        )
    return dataloader