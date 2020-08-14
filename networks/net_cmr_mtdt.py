import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F
from torchvision.models import densenet121, densenet201, resnet50, resnet101, vgg16
import numpy as np


class net_1D(nn.Module):
    """
    Model for metadata
    """
    def __init__(self, num_mtdt):

        super(net_1D, self).__init__()

        self.features = nn.Sequential(
                                     nn.Linear(num_mtdt, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 128),
                                     nn.ReLU()
                                     )

    def forward(self, x):
        x = self.features(x)
        return x



class net_cmr(nn.Module):
    """
    Model for CMR
    """
    def __init__(self, img_size):

        super(net_cmr, self).__init__()

        self.network_3D = resnet50(pretrained=True)

        # New input - 3D volume
        self.network_3D.conv1 = torch.nn.Conv2d(img_size[2], 64, kernel_size=3, padding=1)

        number_ftrs = self.network_3D.fc.in_features

        # Classifier
        self.network_3D.fc = nn.Sequential(
                                      nn.Linear(number_ftrs, 2048),
                                      nn.ReLU()
                                      )

    def forward(self, x):
        x = self.network_3D(x)
        return x



class net_cmr_mtdt(nn.Module):
    """
    Joint-Net
    """

    def __init__(self, args):
        super(net_cmr_mtdt, self).__init__()

        self.cmr_model = net_cmr(args.sax_img_size)
        self.mtdt_model = net_1D(num_mtdt=args.num_mtdt)

        # Combination of features of all branches (i.e. Fundus and mtdt)
        self.combine = nn.Sequential(
                                    nn.Linear(128 + 2048, 2048),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(2048, 1024),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(1024, args.n_classes),
                                    # nn.Sigmoid(),
                                    )


    def forward(self, cmr, mtdt):

        # print(y1.weight.data.cpu().detach().numpy().shape)

        u2 = self.cmr_model(cmr)
        v2 = self.mtdt_model(mtdt)

        # Combining all features from models
        concat_feats = torch.cat((u2,v2), 1)

        # After combining those features, they are later passed through a classifier
        combine = self.combine(concat_feats)

        return combine
