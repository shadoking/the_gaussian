import torch

def InverseSigmoid(x):
    return torch.log(x/(1-x))