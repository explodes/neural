# McCulloch-Pitts neuron model
#
# Evan Leis, 2015
#
# References:
#
# [1] Editor Michael A. Arbib
# "The Handbook of Brain Theory and Neural Networks"
# Cambridge, Massachusetts; London, England: MIT, 2003
#
# [2] http://hyperphysics.phy-astr.gsu.edu/hbase/electronic/xor.html

# Pre-built logic networks

from neural.mcculloch.pitts import model


class XorNetwork(model.Network):
    def __init__(self, a, b):
        super(XorNetwork, self).__init__(*XorNetwork.build_net(a, b))

    @staticmethod
    def build_net(a, b):
        """
        [2] Built as guided by the description provided.
        For the equation of the network, see:
        http://hyperphysics.phy-astr.gsu.edu/hbase/electronic/ietron/xor4.gif
        """
        return model.AndNeuron(
            model.OrNeuron(a, b),
            model.NotNeuron(model.AndNeuron(a, b))
        ),


class HalfAdder(model.Network):
    def __init__(self, a, b):
        super(HalfAdder, self).__init__(*HalfAdder.build_net(a, b))
        self.output = self.outputs[0]
        self.carry = self.outputs[1]

    @staticmethod
    def build_net(a, b):
        sum = XorNetwork.build_net(a, b)[0]
        carry = model.AndNeuron(a, b)
        return sum, carry


class FullAdder(model.Network):
    def __init__(self, cin, a, b):
        super(FullAdder, self).__init__(*FullAdder.build_net(cin, a, b))
        self.output = self.outputs[0]
        self.carry = self.outputs[1]

    @staticmethod
    def build_net(cin, a, b):
        xor = XorNetwork.build_net(a, b)[0]
        sum = XorNetwork.build_net(xor, cin)[0]
        carry = model.OrNeuron(
            model.AndNeuron(xor, cin),
            model.AndNeuron(a, b)
        )
        return sum, carry
