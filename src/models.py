import torch
import numpy as np

class ConvolutionalNeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs,) -> None:
        super().__init__()      # don't need to pass self explicitly when calling the super class
        
        self.embedding = torch.nn.Embedding(num_embeddings=4, embedding_dim=16, padding_idx=0)



