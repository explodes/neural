#!/usr/bin/env python

# McCulloch-Pitts neuron model
#
# Evan Leis, 2015
#
# References:
#
# Editor Michael A. Arbib
# "The Handbook of Brain Theory and Neural Networks"
# Cambridge, Massachusetts; London, England: MIT, 2003

# Make sure things behave like expected

import unittest

from neural.mcculloch.pitts import model


class ModelTestCase(unittest.TestCase):
    def test_or_neuron(self):
        a = model.Input("a")
        b = model.Input("b")
        gate = model.OrNeuron(a, b)

        a.state = 0
        b.state = 0
        gate.update()
        self.assertEquals(gate.state, 0)

        a.state = 1
        b.state = 0
        gate.update()
        self.assertEquals(gate.state, 1)

        a.state = 0
        b.state = 1
        gate.update()
        self.assertEquals(gate.state, 1)

        a.state = 1
        b.state = 1
        gate.update()
        self.assertEquals(gate.state, 1)

    def test_and_neuron(self):
        a = model.Input("")
        b = model.Input("")
        gate = model.AndNeuron(a, b)

        a.state = 0
        b.state = 0
        gate.update()
        self.assertEquals(gate.state, 0)

        a.state = 1
        b.state = 0
        gate.update()
        self.assertEquals(gate.state, 0)

        a.state = 0
        b.state = 1
        gate.update()
        self.assertEquals(gate.state, 0)

        a.state = 1
        b.state = 1
        gate.update()
        self.assertEquals(gate.state, 1)

    def test_not_neuron(self):
        a = model.Input("a")
        gate = model.NotNeuron(a)

        a.state = 0
        gate.update()
        self.assertEquals(gate.state, 1)

        a.state = 1
        gate.update()
        self.assertEquals(gate.state, 0)

    def test_nand_neuron(self):
        a = model.Input("a")
        b = model.Input("b")
        gate = model.NandNeuron(a, b)

        a.state = 0
        b.state = 0
        gate.update()
        self.assertEquals(gate.state, 1)

        a.state = 1
        b.state = 0
        gate.update()
        self.assertEquals(gate.state, 1)

        a.state = 0
        b.state = 1
        gate.update()
        self.assertEquals(gate.state, 1)

        a.state = 1
        b.state = 1
        gate.update()
        self.assertEquals(gate.state, 0)

    def test_simple_net(self):
        """
        a -\
            AND \
        b -/     \
                  AND ->
        c -\     /
            AND /
        d -/
        """
        a = model.Input("a")
        b = model.Input("b")
        g1 = model.AndNeuron(a, b)

        c = model.Input("c")
        d = model.Input("d")
        g2 = model.AndNeuron(c, d)

        final = model.AndNeuron(g1, g2)

        def try_states(_a, _b, _c, _d, expected):
            a.state = _a
            b.state = _b
            g1.update()
            c.state = _c
            d.state = _d
            g2.update()
            final.update()
            self.assertEquals(final.state, expected,
                              "%s should result in %s" %
                              ((_a, _b, _c, _d), expected))

        try_states(0, 0, 0, 0, 0)
        try_states(0, 0, 0, 1, 0)
        try_states(0, 0, 1, 0, 0)
        try_states(0, 0, 1, 1, 0)
        try_states(0, 1, 0, 0, 0)
        try_states(0, 1, 0, 1, 0)
        try_states(0, 1, 1, 0, 0)
        try_states(0, 1, 1, 1, 0)
        try_states(1, 0, 0, 0, 0)
        try_states(1, 0, 0, 1, 0)
        try_states(1, 0, 1, 0, 0)
        try_states(1, 0, 1, 1, 0)
        try_states(1, 1, 0, 0, 0)
        try_states(1, 1, 0, 1, 0)
        try_states(1, 1, 1, 0, 0)
        try_states(1, 1, 1, 1, 1)


class NetworkTestCase(unittest.TestCase):
    def test_gather_inputs_collects_inputs(self):
        a = model.Input("a")
        b = model.Input("b")
        c = model.Input("c")
        d = model.Input("d")
        e = model.Input("e")
        f = model.Input("f")
        g = model.Input("g")
        h = model.Input("h")
        net = model.Network(
            model.AndNeuron(
                model.AndNeuron(a, b),
                model.AndNeuron(c, d)
            ),
            model.AndNeuron(
                model.AndNeuron(e, f),
                model.AndNeuron(g, h)
            )
        )
        self.assertEquals(len(net.inputs), 8)
        self.assertEquals(net.inputs["a"], a)
        self.assertEquals(net.inputs["b"], b)
        self.assertEquals(net.inputs["c"], c)
        self.assertEquals(net.inputs["d"], d)
        self.assertEquals(net.inputs["e"], e)
        self.assertEquals(net.inputs["f"], f)
        self.assertEquals(net.inputs["g"], g)
        self.assertEquals(net.inputs["h"], h)

    def test_getitem(self):
        a = model.Input("a")
        b = model.Input("b")
        c = model.Input("c")
        d = model.Input("d")
        e = model.Input("e")
        f = model.Input("f")
        g = model.Input("g")
        h = model.Input("h")
        and_aa = model.AndNeuron(a, b)
        and_ab = model.AndNeuron(c, d)
        and_a = model.AndNeuron(and_aa, and_ab)
        and_ba = model.AndNeuron(e, f)
        and_bb = model.AndNeuron(g, h)
        and_b = model.AndNeuron(and_ba, and_bb)
        net = model.Network(and_a, and_b)
        self.assertEquals(net["a"], a)
        self.assertEquals(net["b"], b)
        self.assertEquals(net["c"], c)
        self.assertEquals(net["d"], d)
        self.assertEquals(net["e"], e)
        self.assertEquals(net["f"], f)
        self.assertEquals(net["g"], g)
        self.assertEquals(net["h"], h)
        self.assertEquals(net[0], and_a)
        self.assertEquals(net[1], and_b)

    def test_iter_order(self):
        a = model.Input("a")
        b = model.Input("b")
        c = model.Input("c")
        d = model.Input("d")
        e = model.Input("e")
        f = model.Input("f")
        g = model.Input("g")
        h = model.Input("h")
        and_aa = model.AndNeuron(a, b)
        and_ab = model.AndNeuron(c, d)
        and_a = model.AndNeuron(and_aa, and_ab)
        and_ba = model.AndNeuron(e, f)
        and_bb = model.AndNeuron(g, h)
        and_b = model.AndNeuron(and_ba, and_bb)
        net = model.Network(and_a, and_b)

        expected = [a, b, and_aa, c, d, and_ab, and_a, e, f, and_ba, g, h,
                    and_bb, and_b]
        self.assertListEqual(list(net), expected)

    def test_update(self):
        a = model.Input("a")
        b = model.Input("b")
        c = model.Input("c")
        d = model.Input("d")
        e = model.Input("e")
        f = model.Input("f")
        g = model.Input("g")
        h = model.Input("h")
        net = model.Network(
            model.AndNeuron(
                model.AndNeuron(a, b),
                model.AndNeuron(c, d)
            ),
            model.AndNeuron(
                model.AndNeuron(e, f),
                model.AndNeuron(g, h)
            )
        )

        states = net.update(a=1, b=0, c=1, d=0, e=1, f=0, g=1, h=0)
        self.assertEquals(net["a"].state, 1)
        self.assertEquals(net["b"].state, 0)
        self.assertEquals(net["c"].state, 1)
        self.assertEquals(net["d"].state, 0)
        self.assertEquals(net["e"].state, 1)
        self.assertEquals(net["f"].state, 0)
        self.assertEquals(net["g"].state, 1)
        self.assertEquals(net["h"].state, 0)
        self.assertEquals(states, (0, 0))
        states = net.update(a=0, b=1, c=0, d=1, e=0, f=1, g=0, h=1)
        self.assertEquals(net["a"].state, 0)
        self.assertEquals(net["b"].state, 1)
        self.assertEquals(net["c"].state, 0)
        self.assertEquals(net["d"].state, 1)
        self.assertEquals(net["e"].state, 0)
        self.assertEquals(net["f"].state, 1)
        self.assertEquals(net["g"].state, 0)
        self.assertEquals(net["h"].state, 1)
        self.assertEquals(states, (0, 0))
        states = net.update(a=1, b=1, c=1, d=1, e=0, f=0, g=0, h=0)
        self.assertEquals(net["a"].state, 1)
        self.assertEquals(net["b"].state, 1)
        self.assertEquals(net["c"].state, 1)
        self.assertEquals(net["d"].state, 1)
        self.assertEquals(net["e"].state, 0)
        self.assertEquals(net["f"].state, 0)
        self.assertEquals(net["g"].state, 0)
        self.assertEquals(net["h"].state, 0)
        self.assertEquals(states, (1, 0))
        states = net.update(a=0, b=0, c=0, d=0, e=1, f=1, g=1, h=1)
        self.assertEquals(net["a"].state, 0)
        self.assertEquals(net["b"].state, 0)
        self.assertEquals(net["c"].state, 0)
        self.assertEquals(net["d"].state, 0)
        self.assertEquals(net["e"].state, 1)
        self.assertEquals(net["f"].state, 1)
        self.assertEquals(net["g"].state, 1)
        self.assertEquals(net["h"].state, 1)
        self.assertEquals(states, (0, 1))

    def test_can_create_cyclical_network(self):
        a = model.NotNeuron(None)
        a.inputs = [a]

        net = model.Network(a)

        self.assertEqual(list(net), [a])

    def test_cyclical_network_doesnt_explode_during_update(self):
        a = model.NotNeuron(None)
        a.inputs = [a]
        net = model.Network(a)
        net.update()

    def test_cyclical_network_is_self_propelling(self):
        a = model.NotNeuron(None, state=0)
        a.inputs = [a]
        net = model.Network(a)
        self.assertEqual(a.state, 0)
        net.update()
        self.assertEqual(a.state, 1)
        net.update()
        self.assertEqual(a.state, 0)
        net.update()
        self.assertEqual(a.state, 1)
        net.update()
        self.assertEqual(a.state, 0)


if __name__ == '__main__':
    unittest.main()
