import copy

import torch.nn as nn


class DqnAgent:
    def __init__(self, network: nn.Module):
        self.eval_net = network
        self.target_net = copy.deepcopy(network)

    def chose_action(self):
        pass


__all__ = [
    "DqnAgent"
]
