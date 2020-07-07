import torch
import torch.nn as nn

karnel_G = 4
karnel_D = 3
ngf = 32
ndf = 32
nz = 100

class Generator(nn.Module):

    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        # input energy shape [batch x 1 x 1 x 1] going into convolutional
        # self.conv1_1 = nn.ConvTranspose3d(1, 10, karnel_G, 1, 0, bias=False)
        # state size [ ngf*4 x 4 x 4 ]

        # input noise shape [batch x nz x 1 x 1] going into convolutional
        self.conv1_100 = nn.ConvTranspose3d(nz, ngf*8, karnel_G, 1, 0, bias=False)
        # state size [ ngf*4 x 4 x 4 ]


        # outs from first convolutions concatenate state size [ ngf*8 x 4 x 4]
        # and going into main convolutional part of Generator
        self.main_conv = nn.Sequential(

            nn.ConvTranspose3d(ngf*8, ngf*4, karnel_G, 2, 1, bias=False),
            nn.BatchNorm3d(ngf*4),
            nn.ReLU(True),
            # state shape [ (ndf*4) x 8 x 8 ]

            nn.ConvTranspose3d(ngf*4, ngf*2, karnel_G, 2, 1, bias=False),
            nn.BatchNorm3d(ngf*2),
            nn.ReLU(True),
            # state shape [ (ndf*2) x 16 x 16 ]

            nn.ConvTranspose3d(ngf*2, ngf, karnel_G, 2, 1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            # state shape [ (ndf) x 32 x 32 ]

            nn.ConvTranspose3d(ngf, 1, 3, 1, 2, bias=False),
            nn.ReLU()
            # state shape [ 30 x 30 x 30 ]
        )

    def forward(self, noise, energy):
        input = self.conv1_100(noise*energy)
        return self.main_conv(input)




class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        # input shape [30 x 30 x 30] going into convolutional
        self.main_conv = nn.Sequential(

            nn.Conv3d(1, ndf, karnel_D, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state shape [ (ndf) x 32 x 32 ]

            nn.Conv3d(ndf, ndf*2, karnel_D, 2, 1, bias=False),
            nn.BatchNorm3d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state shape [ (ndf*2) x 16 x 16 ]

            nn.Conv3d(ndf*2, ndf*4, karnel_D, 2, 1, bias=False),
            nn.BatchNorm3d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state shape [ (ndf*4) x 8 x 8 ]

            nn.Conv3d(ndf*4, ndf*8, 3, 2, 1, bias=False),
            nn.BatchNorm3d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state shape [ (ndf*8) x 4 x 4 ]

            nn.Conv3d(ndf*8, ndf*4, 3, 2, 0, bias=False)
            # state shape [ 256, 1, 1, 1 ]
        )

        # out shape [ ndf*4, 1, 1, 1 ] from convolutional concatenates with input energy
        # and going into fully conected classifier
        self.fc = nn.Sequential(
            nn.Linear(ndf*4 + 1, ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Linear(ndf*2, ndf),
            # nn.LayerNorm(ndf),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3),

            nn.Linear(ndf*2, 1),
            nn.Sigmoid()
        )

    def forward(self, shower, energy):
        conv_out = self.main_conv(shower)
        fc_input = torch.cat((conv_out, energy), 1).view(-1, ndf*4 + 1)
        return self.fc(fc_input)
