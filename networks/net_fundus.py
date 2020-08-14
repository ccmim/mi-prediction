import torch.nn as nn
from torchvision.models import densenet121, densenet201, resnet50, resnet101, vgg16


class net_fundus(nn.Module):
    """
    Model for CMR
    """
    def __init__(self, args):

        super(net_fundus, self).__init__()

        self.network_3D = resnet50(pretrained=True)

        number_ftrs = self.network_3D.fc.in_features

        # Classifier
        self.network_3D.fc = nn.Sequential(
                                        nn.Linear(number_ftrs, args.n_classes)
                                      )

    def forward(self, x):
        x = self.network_3D(x)
        return x
