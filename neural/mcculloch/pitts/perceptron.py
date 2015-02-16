# -*- coding: utf-8 -*-
#
# McCulloch-Pitts neuron model
#
# Evan Leis, 2015
#
# References:
#
# [1] Editor Michael A. Arbib
# "The Handbook of Brain Theory and Neural Networks"
# Cambridge, Massachusetts; London, England: MIT, 2003

# Scratchpad for the implementation of a neuron that can learn.
# Once it is implemented, the logic should be placed in the standard Neuron
# model so that the Neuron subclasses (logic gates) can learn, too, even though
# the actual usefulness of such is debatable.

from neural.mcculloch.pitts import model


class Perceptron(model.Neuron):
    def train(self, expected_state, learning_rate):
        """
        Update the weights of this neuron to converge to a smarter `Neuron`

        [1] pp. 20

        Δw[ij] = k(Y[i] - y[i])x[j]
        """
        difference = expected_state - self.state
        if difference == 0:
            return
        for index in xrange(len(self.inputs)):
            input = self.inputs[index]
            delta = learning_rate * difference * input.state
            self.weights[index] += delta


class SmartNetwork(model.Network):
    def learn(self, expected_states, learning_rate):
        """
        It's easy to train an individual neuron.
        It's a lot trickier to train a network.

        [1] pp. 20

        Δw[ij] = k(Y[i] - y[i])x[j]
        """
        # TODO: The multi-variable derivatives on page 22 seem a bit outrageous.




