import os
import pickle
import config
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

def load_data():
    """
    Load the training and test data from the specified filepaths.
    Parameters:
        None (defined globally in config.py)
    Returns:
        train_data (pd.DataFrame): The training data.
        test_data (pd.DataFrame): The test data.
    Raises:
        IOError: If the data cannot be loaded from the specified path.
    Example:
        train_data, test_data = load_data()
    """
    new_columns = ["Names", "Sequences", "Labels"]

    # not necessary, but DataFrames are easier to manipulate downstream
    train_data = pd.DataFrame(pd.read_pickle(config.TRAIN_DATA_PATH)).T
    test_data = pd.DataFrame(pd.read_pickle(config.TEST_DATA_PATH)).T

    train_data.columns = new_columns
    test_data.columns = new_columns
    
    print("Data loaded successfully")

    return train_data, test_data

def save_model(model, model_name: str):
    """
    Save a machine learning model to a specified directory.
    Parameters:
        model: The machine learning model to be saved.
        model_name (str): The name of the file to save the model as.
    Returns:
        None
    Raises:
        IOError: If the model cannot be saved to the specified path.
    Example:
        save_model(my_model, 'model')
    """
    model_name = model_name + '.pkl'
    model_path = os.path.join(config.MODEL_SAVE_DIR, model_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model {model_name} saved at {model_path}")

class RNADataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initializes an instance of the RNADataset class.

        This class inherits from the PyTorch Dataset class and is used to load and preprocess RNA sequence data,
        which can then be fed into a PyTorch DataLoader for training a machine learning model. The strings for
        RNA sequences and secondary structures are converted into numerical tensor representations.

        Methods:
            __len__: Returns the number of samples in the dataset.
            __getitem__: Returns the sequence tensor and secondary structure matrix tensor for a given index.

        Parameters:
            dataframe (pd.DataFrame): A pandas dataframe with columns 'Names', 'Sequences', and 'Structures'.
        """
        self.dataframe = dataframe
        self.names = dataframe["Names"]
        self.sequences = dataframe["Sequences"]
        self.structures = dataframe["Structures"]
        
        # do we need to filter this to only include 'AUGC'?
        self.sequence_vocab = {char: idx for idx, char in enumerate(set(''.join(self.sequences)))}
        
        # don't need this, secondary structures are encoded differently
        # self.structure_vocab = {char: idx for idx, char in enumerate(set(''.join(self.structures)))}

    def __len__(self):
        """Returns number of samples in the dataset."""
        return len(self.dataframe)
    
    def __getitem__(self, idx: int):
        """Returns the sequence tensor and secondary structure matrix tensor for a given index."""

        # iloc allows us to index into the dataframe by row, rather than by column name
        sequence = self.sequences.iloc[idx]
        structure = self.structures.iloc[idx]

        # convert the sequence and structure strings into numerical tensors
        sequence_tensor = torch.tensor([self.sequence_vocab[char] for char in sequence], dtype=torch.long)

        structure_matrix = self.convert_structure_to_matrix(sequence, structure)
        structure_tensor = torch.tensor(structure_matrix, dtype=torch.float32)

        # return the sequence tensor and structure tensor, 
        # we could also return the name and other metadata if needed
        return sequence_tensor, structure_tensor
    
    def convert_structure_to_matrix(self, sequence: str, structure: str) -> np.ndarray:
        """
        Constructs a secondary structure matrix for an RNA sequence based on its dot-bracket notation.

        The secondary structure matrix is an N x N matrix (where N is the length of the sequence) that 
        represents base pairings in the secondary structure. A '1' in position (i, j) of the matrix 
        indicates that the nucleotide at position i is paired with the nucleotide at position j, 
        and the matrix is symmetric since RNA base pairings are bidirectional.

        Parameters:
        ----------
        sequence : str
            The RNA sequence (not used directly in the computation but required for matrix size).
        structure : str
            Dot-bracket notation representing the RNA secondary structure. '(' represents a nucleotide
            that is base-paired with a later nucleotide, and ')' represents a nucleotide that is paired
            with a preceding nucleotide. Dots '.' represent unpaired nucleotides.

        Returns:
        -------
        matrix : np.ndarray
            An N x N matrix where N is the length of the sequence, with 1s indicating base pairings 
            and 0s elsewhere.

        Example:
        -------
        sequence = "GCAU"
        structure = "(..)"
        matrix = construct_secondary_structure_matrix(sequence, structure)
        # matrix will be:
        # array([[0, 0, 1, 0],
        #        [0, 0, 0, 0],
        #        [1, 0, 0, 0],
        #        [0, 0, 0, 0]])
        """
        N = len(sequence)
        matrix = np.zeros((N, N), dtype=int)

        # Stack to hold positions of '('
        stack = []

        for i, char in enumerate(structure):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    matrix[i][j] = 1
                    matrix[j][i] = 1  # symmetric pairing, RNA base-pairing is bidirectional

        return matrix
    

def create_dataloader(dataframe: pd.DataFrame, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Creates a PyTorch DataLoader object from a given dataset.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        A pandas DataFrame containing the RNA sequence and structure data.
    batch_size : int
        The batch size for the DataLoader.
    shuffle : bool, default=True
        Whether to shuffle the data in the DataLoader.

    Returns:
    -------
    dataloader : DataLoader
        A PyTorch DataLoader object that can be used to iterate over the dataset in batches.

    Example:
    -------
    train_loader = create_data_loader(train_dataset, batch_size=32, shuffle=True)
    """
    dataset = RNADataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return dataloader

def collate_fn(batch: list):
    """
    Collate function for the RNA dataset DataLoader.
    As the sequences have variable lengths, we need to pad them to create a tensor of equal-sized sequences.

    Parameters:
    ----------
    batch : list
        A list of tuples where each tuple contains (sequence_tensor, structure_tensor).
    """
    # unpack sequences and structure matrices
    sequences, structure_matrices = zip(*batch)

    # pad sequences to the same length
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)

    # structure matrices don't need padding (as they're always square, thank god)
    # so we can just stack them along a new dimension
    structure_matrices_stacked = torch.stack(structure_matrices)

    return sequences_padded, structure_matrices_stacked
