import numpy as np
import torch
from torch import nn

from mousenet.config.config import EDGE_Z


class Conv2dMask(nn.Conv2d):
    """
    Conv2d with Gaussian mask
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        gsh,
        gsw,
        mask=3,
        stride=1,
        padding=0,
    ):
        super(Conv2dMask, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.mypadding = nn.ConstantPad2d(padding, 0)
        if mask == 0:
            self.mask = None
        if mask == 1:
            self.mask = nn.Parameter(
                torch.Tensor(self.make_gaussian_kernel_mask(gsh, gsw))
            )
        elif mask == 2:
            self.mask = nn.Parameter(
                torch.Tensor(self.make_gaussian_kernel_mask(gsh, gsw)),
                requires_grad=False,
            )
        elif mask == 3:
            self.mask = nn.Parameter(
                torch.Tensor(
                    self.make_gaussian_kernel_mask_vary_channel(
                        gsh, gsw, kernel_size, out_channels, in_channels
                    )
                ),
                requires_grad=False,
            )
        else:
            assert "mask should be 0, 1, 2, 3!"

    def forward(self, input):
        if self.mask is not None:
            return super(Conv2dMask, self)._conv_forward(
                self.mypadding(input), self.weight * self.mask, self.bias
            )
        else:
            return super(Conv2dMask, self)._conv_forward(
                self.mypadding(input), self.weight, self.bias
            )

    def make_gaussian_kernel_mask(self, peak, sigma):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param edge_z: Z-score (# standard deviations) of edge of kernel
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        width = int(sigma * EDGE_Z)
        x = np.arange(-width, width + 1)
        X, Y = np.meshgrid(x, x)
        radius = np.sqrt(X**2 + Y**2)

        probability = peak * np.exp(-(radius**2) / 2 / sigma**2)

        re = np.random.rand(len(x), len(x)) < probability
        # plt.imshow(re, cmap='Greys')
        return re

    def make_gaussian_kernel_mask_vary_channel(
        self, peak, sigma, kernel_size, out_channels, in_channels
    ):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param edge_z: Z-score (# standard deviations) of edge of kernel
        :param kernel_size: kernel size of the conv2d
        :param out_channels: number of output channels of the conv2d
        :param in_channels: number of input channels of the con2d
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        re = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        for i in range(out_channels):
            for j in range(in_channels):
                re[i, j, :] = self.make_gaussian_kernel_mask(peak, sigma)
        return re
