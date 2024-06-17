import networkx as nx
import torch
from torch import nn

from mousenet.config.config import OUTPUT_AREAS, SUBFIELDS
from mousenet.model.cnn_layers.Conv2dMask import Conv2dMask
from mousenet.model.subsample_polygon_indeces import get_subsample_indices


class MouseNetCompletePool(nn.Module):
    """
    torch model constructed by parameters provided in network.
    """

    def __init__(self, network, mask=3):
        super(MouseNetCompletePool, self).__init__()
        self.Convs = nn.ModuleDict()
        self.BNs = nn.ModuleDict()
        self.network = network
        self.sub_indices = {}

        G, _ = network.make_graph()
        self.top_sort = list(nx.topological_sort(G))

        # if self.retinomap is not None:
        #     for layer in self.top_sort:
        #         self.layer_masks[layer] = get_retinotopic_mask(layer, self.retinomap)
        # else:
        #     for layer in self.top_sort:
        #         self.layer_masks[layer] = torch.ones(32, 32)

        for layer in network.layers:
            params = layer.params
            self.sub_indices[layer.source_name + layer.target_name] = (
                get_subsample_indices(layer)
            )

            self.Convs[layer.source_name + layer.target_name] = Conv2dMask(
                params.in_channels,
                params.out_channels,
                params.kernel_size,
                params.gsh,
                params.gsw,
                stride=params.stride,
                mask=mask,
                padding=params.padding,
            )
            ## plotting Gaussian mask
            # plt.title('%s_%s_%sx%s'%(e[0].replace('/',''), e[1].replace('/',''), params.kernel_size, params.kernel_size))
            # plt.savefig('%s_%s'%(e[0].replace('/',''), e[1].replace('/','')))
            if layer.target_name not in self.BNs:
                self.BNs[layer.target_name] = nn.BatchNorm2d(params.out_channels)

        print(self.sub_indices)
        # calculate total size output to classifier
        total_size = 0

        for area in OUTPUT_AREAS:
            layer = network.find_conv_source_target("%s2/3" % area[:-1], "%s" % area)
            total_size += int(16 * layer.params.out_channels)
        #     if area =='VISp5':
        #         layer = network.find_conv_source_target('VISp2/3','VISp5')
        #         visp_out = layer.params.out_channels
        #         # create 1x1 Conv downsampler for VISp5
        #         visp_downsample_channels = visp_out
        #         ds_stride = 2
        #         self.visp5_downsampler = nn.Conv2d(visp_out, visp_downsample_channels, 1, stride=ds_stride)
        #         total_size += INPUT_SIZE[1]/ds_stride * INPUT_SIZE[2]/ds_stride * visp_downsample_channels
        #     else:
        #         layer = network.find_conv_source_target('%s2/3'%area[:-1],'%s'%area)
        #         total_size += int(layer.out_size*layer.out_size*layer.params.out_channels)

        # self.classifier = nn.Sequential(
        # nn.Linear(int(total_size), NUM_CLASSES),
        # nn.Linear(int(total_size), HIDDEN_LINEAR),
        # nn.ReLU(True),
        # nn.Dropout(),
        # nn.Linear(HIDDEN_LINEAR, HIDDEN_LINEAR),
        # nn.ReLU(True),
        # nn.Dropout(),
        # nn.Linear(HIDDEN_LINEAR, NUM_CLASSES),
        # )

    def get_img_feature(self, x, area_list, return_calc_graph=False, flatten=False):
        """
        function for get activations from a list of layers for input x
        :param x: input image set Tensor with size (num_img, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
        :param area_list: a list of area names
        :return: if list length is 1, return the (flatten/unflatten) activation of that area
                 if list length is >1, return concatenated flattened activation of the areas.
        """
        calc_graph = {}

        for area in self.top_sort:
            if area == "input":
                continue

            if area == "LGNd" or area == "LGNv":
                layer = self.network.find_conv_source_target("input", area)
                layer_name = layer.source_name + layer.target_name

                if SUBFIELDS:
                    left, width, bottom, height = self.sub_indices[layer_name]
                    # since the input to conv is of shape: (N, C, H, W)
                    source_field = torch.narrow(
                        torch.narrow(x, 3, left, width), 2, bottom, height
                    )  # TODO: check top/bottom direction
                    calc_graph[area] = nn.ReLU(inplace=True)(
                        self.BNs[area](self.Convs[layer_name](source_field))
                    )
                else:
                    calc_graph[area] = nn.ReLU(inplace=True)(
                        self.BNs[area](self.Convs[layer_name](x))
                    )

                continue

            for layer in self.network.layers:
                if layer.target_name == area:
                    # mask = None
                    # if layer.source_name in self.layer_masks:
                    #     mask = self.layer_masks[layer.source_name]
                    # if mask is None:
                    #     mask = 1
                    layer_name = layer.source_name + layer.target_name
                    if SUBFIELDS:
                        left, width, bottom, height = self.sub_indices[
                            layer_name
                        ]  # TODO: incorporate padding here
                        # since the input to conv is of shape: (N, C, H, W)

                        source_field = torch.narrow(
                            torch.narrow(calc_graph[layer.source_name], 3, left, width),
                            2,
                            bottom,
                            height,
                        )
                        layer_output = self.Convs[layer_name](source_field)
                    else:
                        layer_output = self.Convs[layer_name](
                            calc_graph[layer.source_name]
                        )
                    if area not in calc_graph:
                        calc_graph[area] = layer_output
                    else:
                        calc_graph[area] = calc_graph[area] + layer_output
            calc_graph[area] = nn.ReLU(inplace=True)(self.BNs[area](calc_graph[area]))

        # Added for testing
        if return_calc_graph:
            return calc_graph

        if len(area_list) == 1:
            if flatten:
                return torch.flatten(calc_graph["%s" % (area_list[0])], 1)
            else:
                return calc_graph["%s" % (area_list[0])]

        else:
            re = None
            for area in area_list:
                if re is None:
                    re = torch.nn.AdaptiveAvgPool2d(4)(calc_graph[area])
                    # re = torch.flatten(
                    # nn.ReLU(inplace=True)(self.BNs['%s_downsample'%area](self.Convs['%s_downsample'%area](calc_graph[area]))),
                    # 1)
                else:
                    re = torch.cat(
                        [torch.nn.AdaptiveAvgPool2d(4)(calc_graph[area]), re], axis=1
                    )
                    # re=torch.cat([
                    # torch.flatten(
                    # nn.ReLU(inplace=True)(self.BNs['%s_downsample'%area](self.Convs['%s_downsample'%area](calc_graph[area]))),
                    # 1),
                    # re], axis=1)
                # if area == 'VISp5':
                #     re=torch.flatten(self.visp5_downsampler(calc_graph['VISp5']), 1)
                # else:
                #     if re is not None:
                #         re = torch.cat([torch.flatten(calc_graph[area], 1), re], axis=1)
                #     else:
                #         re = torch.flatten(calc_graph[area], 1)
        return re

    def forward(self, x):
        x = self.get_img_feature(x, OUTPUT_AREAS, flatten=False)
        # x = self.classifier(x)
        return x


if __name__ == "__main__":
    from mousenet.constructing_model import load_network_from_pickle

    network = load_network_from_pickle("network_complete_updated_number(3,64,64).pkl")
    mousenet = MouseNetCompletePool(network)
    input = torch.zeros((10, 3, 100, 100))  # first dim batch size
    mousenet.get_img_feature(
        input, ["VISp5", "VISl5", "VISrl5", "VISli5", "VISpl5", "VISal5", "VISpor5"]
    )
