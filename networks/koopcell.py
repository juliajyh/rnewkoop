import torch.nn as nn
import torch
import numpy as np
from spectral_norm_conv_inplace import spectral_norm_conv
from .model_utils import injective_pad, ActNorm2D, Split
from .model_utils import squeeze as Squeeze
from .model_utils import MaxMinGroup
from spectral_norm_fc import spectral_norm_fc

class IresnetBlock(nn.Module):
    def __init__(self, in_shape, int_ch, numTraceSamples=0, numSeriesTerms=0,
                 stride=1, coeff=.97, actnorm=True, n_power_iter=5, nonlin="elu"):

        super(IresnetBlock, self).__init__()
        assert stride in (1, 2)
        self.stride = stride
        self.squeeze = Squeeze(stride)
        self.coeff = coeff
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter
        nonlin = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "softplus": nn.Softplus,
            "sorting": lambda: MaxMinGroup(group_size=2, axis=1)
        }[nonlin]

        # set shapes for spectral norm conv
        in_ch, h, w = in_shape
        inlayers = []
        outlayers = []

        in_ch = in_ch * stride ** 2
        kernel_size1 = 3
        layer1 = nn.Conv2d(in_ch, int_ch, kernel_size=kernel_size1, stride=1, padding=1)
        inlayers.append(spectral_norm_conv(layer1, coeff, (in_ch, h, w), n_power_iterations=self.n_power_iter))

        kernel_size2 = 1
        layer2 = nn.Conv2d(int_ch, int_ch, kernel_size=kernel_size2, padding=0)
        inlayers.append(spectral_norm_fc(layer2, coeff, n_power_iterations=self.n_power_iter))

        kernel_size3 = 1
        layer3 = nn.Conv2d(int_ch, int_ch, kernel_size=kernel_size3, padding=0)
        outlayers.append(spectral_norm_fc(layer3, coeff, n_power_iterations=self.n_power_iter))

        kernel_size4 = 3  # kernel size for first conv
        layer4 = nn.Conv2d(int_ch, in_ch, kernel_size=kernel_size4, padding=1) # zitong: changed the second parameter in_ch -> int_ch
        outlayers.append(spectral_norm_conv(layer4, coeff, (int_ch, h, w), n_power_iterations=self.n_power_iter))

        # inlayers.append(nonlin())
        outlayers.append(nonlin())
        self.inlayers = nn.Sequential(*inlayers)
        self.outlayers = nn.Sequential(*outlayers)
        self.layers = nn.Sequential(*(inlayers + outlayers))
        # if actnorm:
        #     self.actnorm = ActNorm2D(in_ch)
        # else:
        #     self.actnorm = None
        self.actnorm = None

    def forward(self, x, ignore_logdet=False):
        """ bijective or injective block forward """
        if self.stride == 2:
            x = self.squeeze.forward(x)

        # if self.actnorm is not None:
        #     x, an_logdet = self.actnorm(x)
        # else:
        #     an_logdet = 0.0
        an_logdet = 0.0
        # setattr(self.inlayers[0], 'weight', proj)
        Fx = self.inlayers(x)
        # Compute approximate trace for use in training
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)

        # add residual to output
        # y = Fx + x
        return Fx, trace + an_logdet

    def inverse(self, y, old_x, maxIter=30):
        # inversion of ResNet-block (fixed-point iteration)
        y = self.outlayers(y)
        x = y + old_x
        # setattr(self.bottleneck_block[2], 'weight', proj)
        for iter_index in range(maxIter):
            summand = self.layers(x)
            x = x - summand

        if self.actnorm is not None:
            x = self.actnorm.inverse(x)

        # inversion of squeeze (dimension shuffle)
        if self.stride == 2:
            x = self.squeeze.inverse(x)
        return x