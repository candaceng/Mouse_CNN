import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from mousenet.config.config import (
    INPUT_GSH,
    INPUT_GSW,
    INPUT_SIZE,
    KERNEL_SIZE,
    OUTPUT_LGN_MODEL,
    SUBFIELDS,
    get_out_sigma,
)
from mousenet.model.cnn_parameters.ConvLayer import ConvLayer
from mousenet.model.cnn_parameters.ConvParam import ConvParam


class Network:
    """
    network class that contains all conv paramters needed to construct torch model.
    """

    def __init__(self, retinotopic=False):
        self.layers = []
        self.area_channels = {}
        self.area_size = {}
        self.retinotopic = retinotopic

    def find_conv_source_target(self, source_name, target_name):
        for layer in self.layers:
            if layer.source_name == source_name and layer.target_name == target_name:
                return layer
        assert "no conv layer found!"

    def find_conv_target_area(self, target_name):
        for layer in self.layers:
            if layer.target_name == target_name:
                return layer
        assert "no conv layer found!"

    def construct_from_anatomy(self, anet, architecture):
        """
        construct network from anatomy
        :param anet: anatomy class which contains anatomical connections
        :param architecture: architecture class which calls set_num_channels for calculating connection strength
        """
        # construct conv layer for input -> LGNd
        self.area_channels["input"] = INPUT_SIZE[0]
        self.area_size["input"] = OUTPUT_LGN_MODEL[1]

        out_sigma = 1
        # out_channels = np.floor(
        #     anet.find_layer("LGNd", "").num / out_sigma / INPUT_SIZE[1] / INPUT_SIZE[2]
        # )
        out_channels = OUTPUT_LGN_MODEL[0]

        architecture.set_num_channels("LGNd", "", out_channels)
        self.area_channels["LGNd"] = out_channels

        # out_size = INPUT_SIZE[1] * out_sigma

        # # TODO: change hard-coded values
        out_size = (OUTPUT_LGN_MODEL[1] // 2) * out_sigma

        self.area_size["LGNd"] = out_size

        convlayer = ConvLayer(
            "input",
            "LGNd",
            ConvParam(
                in_channels=INPUT_SIZE[0],
                out_channels=out_channels,
                gsh=INPUT_GSH,
                gsw=INPUT_GSW,
                out_sigma=out_sigma,
                kernel_size=KERNEL_SIZE,
            ),
            out_size,
        )

        self.layers.append(convlayer)

        # construct conv layers for all other connections
        G, _ = anet.make_graph()
        Gtop = nx.topological_sort(G)
        root = next(Gtop)  # get root of graph
        for i, e in enumerate(nx.edge_bfs(G, root)):
            in_layer_name = e[0].area + e[0].depth
            out_layer_name = e[1].area + e[1].depth
            print(
                "constructing layer %s: %s to %s" % (i, in_layer_name, out_layer_name)
            )

            in_conv_layer = self.find_conv_target_area(in_layer_name)
            in_size = in_conv_layer.out_size
            in_channels = in_conv_layer.params.out_channels
            # if not isinstance(in_size, int) and not isinstance(in_size, float):  # for LGN
            #     in_channels = in_size[0]
            #     in_size = in_size[1]

            out_anat_layer = anet.find_layer(e[1].area, e[1].depth)

            if self.retinotopic:
                out_sigma = np.sqrt(
                    self.calculate_pixel_area_source_target_ratio(
                        architecture, in_layer_name, out_layer_name
                    )
                )
            else:
                out_sigma = get_out_sigma(e[0].area, e[0].depth, e[1].area, e[1].depth)
            out_size = in_size * out_sigma
            self.area_size[e[1].area + e[1].depth] = out_size

            if SUBFIELDS:
                pixel_area = self.calculate_pixel_area_with_visual_field(
                    architecture, e[1].area
                )
                out_channels = np.floor(out_anat_layer.num / pixel_area)
            else:
                out_channels = np.floor(out_anat_layer.num / out_size**2)

            architecture.set_num_channels(e[1].area, e[1].depth, out_channels)
            self.area_channels[e[1].area + e[1].depth] = out_channels
            convlayer = ConvLayer(
                in_layer_name,
                out_layer_name,
                ConvParam(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    gsh=architecture.get_kernel_peak_probability(
                        e[0].area, e[0].depth, e[1].area, e[1].depth
                    ),
                    gsw=architecture.get_kernel_width_pixels(
                        e[0].area, e[0].depth, e[1].area, e[1].depth
                    ),
                    out_sigma=out_sigma,
                ),
                out_size,
            )
            # convlayer.params.kernel_size = 5
            print(
                f"{in_layer_name} - {out_layer_name}, kernel_size: {convlayer.params.kernel_size}"
            )

            self.layers.append(convlayer)

    def make_graph(self):
        """
        produce networkx graph
        """
        G = nx.DiGraph()
        edges = [(p.source_name, p.target_name) for p in self.layers]
        for edge in edges:
            G.add_edge(edge[0], edge[1])
        node_label_dict = {
            layer: "%s\n%s" % (layer, int(self.area_channels[layer]))
            for layer in G.nodes()
        }
        return G, node_label_dict

    def draw_graph(self, node_size=2000, node_color="yellow", edge_color="red"):
        """
        draw the network structure
        """
        G, node_label_dict = self.make_graph()
        edge_label_dict = {
            (c.source_name, c.target_name): (c.params.kernel_size) for c in self.layers
        }
        plt.figure(figsize=(12, 12))
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        nx.draw(
            G,
            pos,
            node_size=node_size,
            node_color=node_color,
            edge_color=edge_color,
            alpha=0.4,
        )
        nx.draw_networkx_labels(
            G,
            pos,
            node_label_dict,
            font_size=10,
            font_weight=640,
            alpha=0.7,
            font_color="black",
        )
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_label_dict,
            font_size=20,
            font_weight=640,
            alpha=0.7,
            font_color="red",
        )
        plt.show()

    def calculate_pixel_area_with_visual_field(self, architecture, area):
        """
        Calculates the area (in pixels) of an area for a given architecture
        """
        x1, y1, x2, y2 = architecture.get_visual_field_shape(area)
        area = (x2 - x1) * (y2 - y1)
        return area

    def calculate_pixel_area_source_target_ratio(
        self, architecture, source_area, target_area
    ):
        """
        Calculates the ratio of source to target areas, in pixel space
        """
        in_layer_pixel_area = self.calculate_pixel_area_with_visual_field(
            architecture, source_area
        )
        out_layer_pixel_area = self.calculate_pixel_area_with_visual_field(
            architecture, target_area
        )
        return out_layer_pixel_area / in_layer_pixel_area
