# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import cv2
import torch


class FaceDataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.train:
            img_path = self.data.iloc[index]['path']
            image = cv2.imread(img_path)
            labels = self.data.iloc[index]['class']
        else:
            img_path = self.data.iloc[index]['path']
            image = cv2.imread(img_path)
            labels = None


        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]


        return image, labels.astype(np.float32)

def get_dataloader(config):

    batch_size = config.batch_size
    transform_train = config.train.transform
    transform_test = config.test.transform
    csv_file_train = config.train.csv_file
    csv_file_test = config.test.csv_file

    dataset_train = FaceDataset(csv_file_train, train=True, transform=transform_train)
    dataset_test = FaceDataset(csv_file_test, train=False, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=batch_size,
                                        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)
    return {'train_loader': train_loader, 'test_loader': test_loader,
            'dataset_train':dataset_train,
            'dataset_test': dataset_test}


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
