from mousenet.config.config import EDGE_Z


class ConvParam:
    def __init__(
        self, in_channels, out_channels, gsh, gsw, out_sigma, kernel_size=None
    ):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param gsh: Gaussian height for generating Gaussian mask
        :param gsw: Gaussian width for generating Gaussian mask
        :param out_sigma: ratio between output size and input size, 1/2 means reduce output size to 1/2 of the input size
        """

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.gsh = gsh
        self.gsw = gsw
        self.kernel_size = (
            kernel_size if kernel_size is not None else 2 * int(self.gsw * EDGE_Z) + 1
        )

        # KmS = int(self.kernel_size - 1)  # int((self.kernel_size-1/out_sigma))
        # if np.mod(KmS, 2) == 0:
        #     padding = int(KmS / 2)
        # else:
        #     padding = (int(KmS / 2), int(KmS / 2 + 1), int(KmS / 2), int(KmS / 2 + 1))
        self.padding = 0  # padding
        self.stride = 1  # max(1,int(1/out_sigma))
