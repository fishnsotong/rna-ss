import torch
import argparse
import os

from utils import load_data, create_dataloader
from model import ResNet

def parse_args():
    parser = argparse.ArgumentParser(description="RNA-SS Training Script")
    return parser.parse_args()

def train():
    pass

def test():
    pass

def main():
    args = parse_args()
    
    # GPU if availble, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the data (train, test)
    train_df, test_df = load_data()

    # create dataloaders
    train_dataloader = create_dataloader(train_df)
    test_dataloader = create_dataloader(test_df, shuffle=False)

    # TODO: implement the training pipeline

if __name__ == '__main__':
    main()
