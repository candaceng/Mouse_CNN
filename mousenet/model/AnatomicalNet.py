import matplotlib.pyplot as plt
import networkx as nx


class AnatomicalLayer:
    def __init__(self, area, depth, num):
        self.area = area
        self.depth = depth
        self.num = num


class Projection:
    def __init__(self, pre, post):
        self.pre = pre
        self.post = post


class AnatomicalNet:
    def __init__(self):
        self.layers = []
        self.projections = []

    def find_layer(self, area, depth):
        for l in self.layers:
            if l.area == area and l.depth == depth:
                return l
        return None

    def find_projection(self, source_area, source_depth, target_area, target_depth):
        for proj in self.projections:
            if (
                proj.pre.area == source_area
                and proj.pre.depth == source_depth
                and proj.post.area == target_area
                and proj.post.depth == target_depth
            ):
                return proj
        return None

    def add_layer(self, layer):
        assert isinstance(layer, AnatomicalLayer)
        if self.find_layer(layer.area, layer.depth):
            print("%s %s already exist!" % (layer.area, layer.depth))
            return
        self.layers.append(layer)

    def add_projection(self, projection):
        assert isinstance(projection, Projection)
        if self.find_projection(
            projection.pre.area,
            projection.pre.depth,
            projection.post.area,
            projection.post.depth,
        ):
            print(
                "Projection %s %s to %s %s already exist!"
                % (
                    projection.pre.area,
                    projection.pre.depth,
                    projection.post.area,
                    projection.post.depth,
                )
            )
            return
        self.projections.append(projection)

    def make_graph(self):
        G = nx.DiGraph()
        edges = [(p.pre, p.post) for p in self.projections]
        for edge in edges:
            G.add_edge(edge[0], edge[1])
        node_label_dict = {layer: layer.area + layer.depth for layer in G.nodes()}
        return G, node_label_dict

    def draw_graph(self, node_size=1600, node_color="yellow", edge_color="red"):
        G, node_label_dict = self.make_graph()
        plt.figure(figsize=(10, 10))
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        nx.draw(
            G, pos, node_size=node_size, node_color=node_color, edge_color=edge_color
        )
        nx.draw_networkx_labels(G, pos, node_label_dict)
        plt.show()
