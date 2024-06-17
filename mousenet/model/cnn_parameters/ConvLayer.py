class ConvLayer:
    def __init__(self, source_name, target_name, params, out_size):
        """
        :param params: ConvParam containing the parameters of the layer
        :param source_name: name of the source area, e.g. VISp4, VISp2/3, VISp5
        :param target_name: name of the target area
        :param out_size: output size of the layer
        """
        self.params = params
        self.source_name = source_name
        self.target_name = target_name
        self.out_size = out_size
