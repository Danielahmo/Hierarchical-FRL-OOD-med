import math
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        super(Encoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        n = math.log2(isize)
        assert n == round(n), "imageSize must be a power of 2"
        n = int(n)

        main = nn.Sequential()
        main.add_module("input-conv", nn.Conv3d(nc, ngf, 4, 2, 1, bias=False))
        main.add_module("input-BN", nn.BatchNorm3d(ngf))
        main.add_module("input-relu", nn.ReLU(True))

        for i in range(n - 3):
            # state size. (ngf) x 32 x 32
            main.add_module(
                "pyramid:{0}-{1}:conv".format(ngf * 2**i, ngf * 2 ** (i + 1)),
                nn.Conv3d(ngf * 2 ** (i), ngf * 2 ** (i + 1), 4, 2, 1, bias=False),
            )
            main.add_module(
                "pyramid:{0}:batchnorm".format(ngf * 2 ** (i + 1)),
                nn.BatchNorm3d(ngf * 2 ** (i + 1)),
            )
            main.add_module(
                "pyramid:{0}:relu".format(ngf * 2 ** (i + 1)), nn.ReLU(True)
            )
        self.conv1 = nn.Conv3d(ngf * 2 ** (n - 3), nz, 4)
        self.conv2 = nn.Conv3d(ngf * 2 ** (n - 3), nz, 4)

        self.main = main

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def forward(self, input):

        output = self.main(input)
      
        mu = self.conv1(output)
        logvar = self.conv2(output)
        z = self.reparametrize(mu, logvar)
        return [z, mu, logvar]


class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, slices=None):
        super(DCGAN_G, self).__init__()
        self.nc = nc
        self.isize = isize
        if slices == None:
            self.slices = isize
        else:
            self.slices = slices

        assert isize % 16 == 0 and self.slices % 16 ==0, "isize has to be a multiple of 16"

        cngf, tisize, tslices = ngf // 2, 4, 4
        while tisize < isize:
            cngf *= 2
            tisize *= 2
        while tslices < self.slices:
            cngf *= 2
            tslices *= 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module(
            "initial:{0}-{1}:convt".format(nz, cngf),
            nn.ConvTranspose3d(nz, cngf, 4, 1, 0, bias=False),
        )
        main.add_module("initial:{0}:batchnorm".format(cngf), nn.BatchNorm3d(cngf))
        main.add_module("initial:{0}:relu".format(cngf), nn.ReLU(True))

        csize = 4
        while csize < isize // 2:
            main.add_module(
                "pyramid:{0}-{1}:convt".format(cngf, cngf // 2),
                nn.ConvTranspose3d(cngf, cngf // 2, 4, 2, 1, bias=False),
            )
            main.add_module(
                "pyramid:{0}:batchnorm".format(cngf // 2), nn.BatchNorm3d(cngf // 2)
            )
            main.add_module("pyramid:{0}:relu".format(cngf // 2), nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module(
            "final:{0}-{1}:convt".format(cngf, nc),
            nn.ConvTranspose3d(cngf, nc , 4, 2, 1, bias=False),
        )

        self.main = main

    def forward(self, z):
        x = self.main(z)
        return x  # Shape: [B, C, D, H, W]