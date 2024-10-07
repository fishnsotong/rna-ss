from utils import load_data, create_dataloader

def main():

    # load the data (train, test)
    train_df, test_df = load_data()

    # create dataloaders
    train_dataloader = create_dataloader(train_df)
    test_dataloader = create_dataloader(test_df, shuffle=False)

    # TODO: implement the training pipeline

    pass

if __name__ == '__main__':
    main()