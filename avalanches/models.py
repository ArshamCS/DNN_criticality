#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import argparse

class MLPWithPerturbationStop(nn.Module):
    """
    Multilayer perceptron (MLP) with early stopping:
    propagation stops when the second moment of activations falls below threshold.
    All layers are standard Linear layers with tanh activations.
    """
    def __init__(self, N, L_max, sigma_w2, sigma_b2):
        super().__init__()
        self.N = N
        self.L_max = L_max

        # All layers, including the first, treated the same
        self.layers = nn.ModuleList([
            nn.Linear(N, N) for _ in range(L_max)
        ])

        # Initialize weights and biases
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=np.sqrt(sigma_w2)/ np.sqrt(N))
            nn.init.normal_(layer.bias, mean=0.0, std=np.sqrt(sigma_b2))

    def forward(self, x, threshold=1):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # ensure batch dimension

        avalanches = []
        for layer in self.layers:
            avalanches.append(np.sqrt(x.pow(2).sum().item()))
            x_tanh = torch.tanh(x)
            x = layer(x_tanh)
            if x.pow(2).sum().item() < threshold:
                avalanches.append(np.sqrt(x.pow(2).sum().item()))
                break

        return [] , avalanches
