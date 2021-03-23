import torch.nn as nn


class net_mtdt(nn.Module):
    """
    Model for metadata
    """
    def __init__(self, args):

        super(net_mtdt, self).__init__()

        self.features = nn.Sequential(
                                     nn.Linear(args.num_mtdt, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, args.n_classes),
                                     )

    def forward(self, x):
        x = self.features(x)
        return x
