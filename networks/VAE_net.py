"""
Networks created by Andres Diaz-Pinto
"""

import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self, out_size):
        super(Flatten, self).__init__()
        self.out_size = out_size
    def forward(self, input):
        return input.view(-1, self.out_size)

class LinearDec(nn.Module):
    def __init__(self, out_size):
        super(LinearDec, self).__init__()
        self.out_size = out_size
    def forward(self, input):
        return input.view(-1, self.out_size[0], self.out_size[1], self.out_size[2])


################# NETS FOR CMR IMAGES

class encoder_cmr(nn.Module):
    def __init__(self, image_channels, ndf, z_dim):
        super(encoder_cmr, self).__init__()

        self.image_channels = image_channels
        self.ndf = ndf
        self.z_dim = z_dim

        self.encoder = nn.Sequential(

            nn.Conv2d(self.image_channels, self.ndf, 4, 2, 1),
            nn.BatchNorm2d(self.ndf),
            nn.ReLU(),

            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*2),
            nn.ReLU(),

            nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*4),
            nn.ReLU(),

            nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*8),
            nn.ReLU(),

            nn.Conv2d(self.ndf*8, self.ndf*16, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*16),
            nn.ReLU(),

            Flatten(out_size = self.ndf*16*4*4),

            nn.Linear(self.ndf*16*4*4, self.z_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z


class decoder_cmr(nn.Module):
    def __init__(self, image_channels, ndf, z_dim):
        super(decoder_cmr, self).__init__()

        self.image_channels = image_channels
        self.ndf = ndf
        self.z_dim = z_dim

        self.decoder = nn.Sequential(

            nn.Linear(self.z_dim, self.ndf*16*4*4),
            nn.LeakyReLU(0.2),

            LinearDec(out_size = [self.ndf*16, 4, 4]),

            nn.ConvTranspose2d(self.ndf*16, self.ndf*8, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(self.ndf*8, self.ndf*4, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(self.ndf*4, self.ndf*2, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(self.ndf*2, self.ndf, 4, 2, 1),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(self.ndf, self.image_channels, 4, 2, 1),
            nn.Tanh(),
            )

    def forward(self, x):
        h = self.decoder(x)
        return h


################# NETS FOR FUNDUS IMAGES

class encoder_fundus(nn.Module):
    def __init__(self, image_channels, ndf, z_dim):
        super(encoder_fundus, self).__init__()

        self.image_channels = image_channels
        self.ndf = ndf
        self.z_dim = z_dim

        self.encoder = nn.Sequential(

            nn.Conv2d(self.image_channels, self.ndf, 4, 2, 1),
            nn.BatchNorm2d(self.ndf),
            nn.ReLU(),

            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*2),
            nn.ReLU(),

            nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*4),
            nn.ReLU(),

            nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*8),
            nn.ReLU(),

            nn.Conv2d(self.ndf*8, self.ndf*16, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*16),
            nn.ReLU(),

            Flatten(out_size = self.ndf*16*4*4),

            nn.Linear(self.ndf*16*4*4, self.z_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z


class decoder_fundus(nn.Module):
    def __init__(self, image_channels, ndf, z_dim):
        super(decoder_fundus, self).__init__()

        self.image_channels = image_channels
        self.ndf = ndf
        self.z_dim = z_dim

        self.decoder = nn.Sequential(

            nn.Linear(self.z_dim, self.ndf*16*4*4),
            nn.LeakyReLU(0.2),

            LinearDec(out_size = [self.ndf*16, 4, 4]),

            nn.ConvTranspose2d(self.ndf*16, self.ndf*8, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(self.ndf*8, self.ndf*4, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(self.ndf*4, self.ndf*2, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(self.ndf*2, self.ndf, 4, 2, 1),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(self.ndf, self.image_channels, 4, 2, 1),
            nn.Tanh(),
            )

    def forward(self, x):
        h = self.decoder(x)
        return h



