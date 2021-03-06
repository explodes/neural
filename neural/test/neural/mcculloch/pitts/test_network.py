# McCulloch-Pitts neuron model
#
# Evan Leis, 2015

# Make sure things behave like expected

import unittest

from neural.mcculloch.pitts import model, network


class XorNetworkTestCase(unittest.TestCase):
    def test_xor_network(self):
        a = model.Input("a")
        b = model.Input("b")
        xor = network.XorNetwork(a, b)

        state = xor.update(a=0, b=0)
        self.assertEqual(state, (0,))

        state = xor.update(a=1, b=1)
        self.assertEqual(state, (0,))

        state = xor.update(a=1, b=0)
        self.assertEqual(state, (1,))

        state = xor.update(a=0, b=1)
        self.assertEqual(state, (1,))


class HalfAdderTestCase(unittest.TestCase):
    def test_half_adder(self):
        a = model.Input("a")
        b = model.Input("b")
        half_adder = network.HalfAdder(a, b)

        state = half_adder.update(a=0, b=0)
        self.assertEqual(state, (0, 0))
        self.assertEqual(half_adder.output.state, 0)
        self.assertEqual(half_adder.carry.state, 0)

        state = half_adder.update(a=1, b=1)
        self.assertEqual(state, (0, 1))
        self.assertEqual(half_adder.output.state, 0)
        self.assertEqual(half_adder.carry.state, 1)

        state = half_adder.update(a=1, b=0)
        self.assertEqual(state, (1, 0))
        self.assertEqual(half_adder.output.state, 1)
        self.assertEqual(half_adder.carry.state, 0)

        state = half_adder.update(a=0, b=1)
        self.assertEqual(state, (1, 0))
        self.assertEqual(half_adder.output.state, 1)
        self.assertEqual(half_adder.carry.state, 0)


class FullAdderTestCase(unittest.TestCase):
    def test_full_adder(self):
        cin = model.Input("cin")
        a = model.Input("a")
        b = model.Input("b")
        full_adder = network.FullAdder(cin, a, b)

        state = full_adder.update(cin=0, a=0, b=0)
        self.assertEqual(state, (0, 0))
        self.assertEqual(full_adder.output.state, 0)
        self.assertEqual(full_adder.carry.state, 0)

        state = full_adder.update(cin=0, a=1, b=1)
        self.assertEqual(state, (0, 1))
        self.assertEqual(full_adder.output.state, 0)
        self.assertEqual(full_adder.carry.state, 1)

        state = full_adder.update(cin=0, a=1, b=0)
        self.assertEqual(state, (1, 0))
        self.assertEqual(full_adder.output.state, 1)
        self.assertEqual(full_adder.carry.state, 0)

        state = full_adder.update(cin=0, a=0, b=1)
        self.assertEqual(state, (1, 0))
        self.assertEqual(full_adder.output.state, 1)
        self.assertEqual(full_adder.carry.state, 0)

        state = full_adder.update(cin=1, a=0, b=0)
        self.assertEqual(state, (1, 0))
        self.assertEqual(full_adder.output.state, 1)
        self.assertEqual(full_adder.carry.state, 0)

        state = full_adder.update(cin=1, a=1, b=1)
        self.assertEqual(state, (1, 1))
        self.assertEqual(full_adder.output.state, 1)
        self.assertEqual(full_adder.carry.state, 1)

        state = full_adder.update(cin=1, a=1, b=0)
        self.assertEqual(state, (0, 1))
        self.assertEqual(full_adder.output.state, 0)
        self.assertEqual(full_adder.carry.state, 1)

        state = full_adder.update(cin=1, a=0, b=1)
        self.assertEqual(state, (0, 1))
        self.assertEqual(full_adder.output.state, 0)
        self.assertEqual(full_adder.carry.state, 1)
