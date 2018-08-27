import tensorflow as tf

class GannModule:

    def __init__(self, net, index, input_var, in_node_count, out_node_count):
        self.net = net
        self.index = index
        self.index = input_var
        self.in_node_count = in_node_count
        self.out_node_count = out_node_count
        self.build_module()

    def build_module(self):
        pass
