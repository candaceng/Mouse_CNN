import os
import pickle

from mousenet.model.AnatomicalNet import AnatomicalLayer, AnatomicalNet, Projection
from mousenet.model.Network import Network


def gen_network_from_anatomy(architecture):
    anet = gen_anatomy(architecture)
    net = Network()
    net.construct_from_anatomy(anet, architecture)
    return net


def gen_anatomy(
    architecture,
    input_depths=["4"],
    output_depths=["4", "2/3", "5"],
    laminar_connections=[("4", "2/3"), ("2/3", "5")],
):
    """
    generate anatomy structure from data class
    """
    anet = AnatomicalNet()
    areas = architecture.get_areas()
    depths = architecture.get_layers()
    output_map = {}  # collect output layers for each hierarchy

    # create LGNd
    hierarchy = 0
    layer0 = AnatomicalLayer("LGNd", "", architecture.get_num_neurons("LGNd", None))
    anet.add_layer(layer0)
    output_map[0] = [layer0]

    for hierarchy in [1, 2, 3]:
        output_map[hierarchy] = []
        for area in areas:
            if architecture.get_hierarchical_level(area) == hierarchy:
                # create anatomical module for one area
                # add layers
                area_layers = {}
                for depth in depths:
                    layer = AnatomicalLayer(
                        area, depth, architecture.get_num_neurons(area, depth)
                    )
                    area_layers[depth] = layer
                    anet.add_layer(layer)
                    if depth in output_depths:
                        output_map[hierarchy].append(layer)

                # add LaminarProjection
                for source, target in laminar_connections:
                    anet.add_projection(
                        Projection(area_layers[source], area_layers[target])
                    )

                # add AreaProjection
                for depth in depths:
                    if depth in input_depths:
                        for l in output_map[hierarchy - 1]:
                            anet.add_projection(Projection(l, area_layers[depth]))
                        if hierarchy == 3:
                            for l in output_map[hierarchy - 2]:
                                anet.add_projection(Projection(l, area_layers[depth]))
    return anet


def save_network_to_pickle(net, file_path):
    f = open(file_path, "wb")
    pickle.dump(net, f)


def load_network_from_pickle(file_path):
    f = open(file_path, "rb")
    net = pickle.load(f)
    return net


def gen_network(net_name, architecture):
    file_path = "./myresults/%s.pkl" % net_name
    # if os.path.exists(file_path):
    if False:
        net = load_network_from_pickle(file_path)
    else:
        net = gen_network_from_anatomy(architecture)
        if not os.path.exists("./myresults"):
            os.mkdir("./myresults")
        save_network_to_pickle(net, file_path)
    return net
