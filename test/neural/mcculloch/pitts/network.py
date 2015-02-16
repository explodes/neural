#!/usr/bin/env python

# McCulloch-Pitts neuron model
#
# Evan Leis, 2015
#
# References:
#
# Editor Michael A. Arbib "The Handbook of Brain Theory and Neural Networks"
# Cambridge, Massachusetts; London, England: MIT, 2003

# Make sure things behave like expected



from __future__ import absolute_import

import unittest

from neural.mcculloch.pitts import model

from neural.mcculloch.pitts import network


class XorNetworktests(unittest.TestCase):
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


class HalfAdderTests(unittest.TestCase):
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


class FullAdderTests(unittest.TestCase):
    def test_half_adder(self):
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


if __name__ == '__main__':
    unittest.main()