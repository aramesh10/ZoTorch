import torch
from torch import nn

def weight_gradient(layer, input, error):  # a Linear's or Embedding's weight gradient from the error at its output
    rows, columns = layer.weight.shape
    if isinstance(layer, nn.Embedding):
        gradient = torch.zeros_like(layer.weight, dtype=torch.float32)
        return gradient.index_add_(0, input.reshape(-1), error.reshape(-1, columns).float())
    return error.reshape(-1, rows).float().T @ input.reshape(-1, columns).float()
