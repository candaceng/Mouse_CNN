from typing import Tuple

import torch
import torch.nn as nn
from sympy.abc import x as symbolic_x
from sympy.abc import y as symbolic_y

from bmtk.simulator.filternet.lgnmodel.cellmodel import TwoSubfieldLinearCell
from bmtk.simulator.filternet.lgnmodel.linearfilter import SpatioTemporalFilter
from bmtk.simulator.filternet.lgnmodel.spatialfilter import GaussianSpatialFilter
from bmtk.simulator.filternet.lgnmodel.temporalfilter import TemporalFilterCosineBump
from bmtk.simulator.filternet.lgnmodel.transferfunction import MultiTransferFunction


from mousenet.model.lgn_model.FilterParams import FilterParams


class LGNConv3D(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        filter_params: FilterParams,
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.num_neurons = out_channels
        self.filter_params = filter_params

        self.amplitude = self._set_amplitude()
        self.weight = nn.Parameter(
            self._create_kernels(in_channels), requires_grad=False
        )

    def _get_filter_properties(self, is_dominant: bool):
        return self.filter_params.get_filter_properties(is_dominant)

    def _get_spatial_size(self):
        return self.filter_params.spatial_size

    def _get_sfsep_angle(self):
        return self.filter_params.sf_sep, self.filter_params.tuning_angle

    def _set_amplitude(self):
        if not self.filter_params.is_dominant:
            return 1.0
        return (
            -1.0
            if "sOFF" in self.filter_params.model_id
            or "tOFF" in self.filter_params.model_id
            else 1.0
        )

    def _create_kernels(self, in_channels: int):
        """
        kernel data should be of the shape:
         (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])

        :param in_channels:
        :return:
        """
        kernels_data = torch.empty((self.num_neurons, in_channels, *self.kernel_size))
        for i in range(self.num_neurons):
            kernel = self._create_kernel()
            spatiotemporal_kernel = kernel.get_spatiotemporal_kernel(
                range(self.kernel_size[1]), range(self.kernel_size[2])
            )
            # TODO: what is ds (down sampling) rate?
            temporal_ds_rate = (spatiotemporal_kernel.full().shape[0] - 2) // (
                self.kernel_size[0] - 1
            )
            kernels_data[i] = torch.Tensor(spatiotemporal_kernel.full())[
                ::temporal_ds_rate
            ].repeat(in_channels, 1, 1, 1)
        return kernels_data

    def _create_kernel(self):
        dom_kernel = self._create_spatiotemporal_kernel()
        if self.filter_params.is_dominant:
            return dom_kernel
        nondom_kernel = self._create_spatiotemporal_kernel(is_dominant=False)
        sf_sep, angle = self._get_sfsep_angle()
        return TwoSubfieldLinearCell(
            dominant_filter=dom_kernel,
            nondominant_filter=nondom_kernel,
            subfield_separation=sf_sep,
            onoff_axis_angle=angle,
            dominant_subfield_location=(0, 0),
            transfer_function=MultiTransferFunction(
                (symbolic_x, symbolic_y), "Heaviside(x)*(x)+Heaviside(y)*(y)"
            ),
        )

    def _create_spatiotemporal_kernel(self, is_dominant=True):
        weights, kpeaks, delays = self._get_filter_properties(is_dominant)
        sigma = self._get_spatial_size()
        temporal_filter = TemporalFilterCosineBump(
            weights=weights, kpeaks=kpeaks, delays=delays
        )
        spatial_filter = GaussianSpatialFilter((0.0, 0.0), (sigma, sigma), 0, "center")
        return SpatioTemporalFilter(
            spatial_filter, temporal_filter, self.amplitude
        )  # TODO: additional parameters: threshold: int, reverse: bool
