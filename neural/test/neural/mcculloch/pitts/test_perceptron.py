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

from neural.mcculloch.pitts import model, perceptron


class PerceptronTestCase(unittest.TestCase):
    def test_basic_perceptron(self):
        a = model.Input("a")
        b = model.Input("b")
        c = model.Input("c")
        d = model.Input("d")
        p = perceptron.Perceptron((a, b, c, d), (0, 0, 0, 0), 1)
        n = model.Network(p)

        def get_expected(a_in, b_in, c_in, d_in):
            # The output should be true iff a_in is true and b_in is true
            return 1 if a_in == 1 and b_in == 1 else 0

        def learn(a_in, b_in, c_in, d_in, batch_size=2500, learning_rate=0.25):
            e = get_expected(a_in, b_in, c_in, d_in)
            for dummy in xrange(batch_size):
                n.update(a=a_in, b=b_in, c=c_in, d=d_in)
                p.train(e, learning_rate)

        def test(a_in, b_in, c_in, d_in):
            state, = n.update(a=a_in, b=b_in, c=c_in, d=d_in)
            return state

        # without learning, the network is accurate 12.5% of the time
        for _a in (0, 1):
            for _b in (0, 1):
                for _c in (0, 1):
                    for _d in (0, 1):
                        learn(_a, _b, _c, _d)
                        pass

        n_tests = 0
        n_successes = 0

        for _a in (0, 1):
            for _b in (0, 1):
                for _c in (0, 1):
                    for _d in (0, 1):
                        n_tests += 1
                        value = test(_a, _b, _c, _d)
                        expected = get_expected(_a, _b, _c, _d)
                        if value == expected:
                            n_successes += 1

        self.assertEqual(n_tests, n_successes)
        self.assertEqual(p.weights, [0.5, 0.5, 0, 0])
