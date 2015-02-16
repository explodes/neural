# McCulloch-Pitts neuron model
#
# Evan Leis, 2015
#
# References:
#
# [1] Editor Michael A. Arbib "The Handbook of Brain Theory and Neural Networks"
# Cambridge, Massachusetts; London, England: MIT, 2003

# The example modelling is to imitate a wired connection between inputs,
# neurons, and outputs.


class Input:
    """
    Input gate/wire.
    """

    def __init__(self, name=None, state=0):
        self.name = name
        self.state = state

    def __repr__(self):
        return u"Input(name=%r, state=%r)" % (self.name, self.state)


class Neuron(object):
    """
    McCulloch-Pitts Neuron

    [1] pp.8
    """

    def __init__(self, inputs, weights, threshold, state=0):
        """
        Create a neuron with the given input wires, weights for each respective
        wire, neuron threshold, and initial output state.
        """
        self.inputs = inputs
        self.weights = weights
        self.threshold = threshold
        self.state = state

    def update(self):
        """
        Update the output state based on the state of the input

        y(t+1) = 1 iff sum(wi * xi(t)) >= threshold
        """
        value = 0
        for index in xrange(len(self.inputs)):
            value += self.inputs[index].state * self.weights[index]
        self.state = 1 if value >= self.threshold else 0

    def __repr__(self):
        return u"Neuron(%r, %r, %r, state=%r)" % (
            self.inputs, self.weights, self.threshold, self.state)


class AndNeuron(Neuron):
    """
    AND
    a * 1 + b * 1 >= 2
    """

    def __init__(self, a, b, state=0):
        super(AndNeuron, self).__init__((a, b), (1, 1), 2, state=state)


class OrNeuron(Neuron):
    """
    OR
    a * 1 + b * 1 >= 1
    """

    def __init__(self, a, b, state=0):
        super(OrNeuron, self).__init__((a, b), (1, 1), 1, state=state)


class NotNeuron(Neuron):
    """
    NOT
    a * -1 >= -1
    """

    def __init__(self, a, state=0):
        super(NotNeuron, self).__init__((a,), (-1,), 0, state=state)


class NandNeuron(Neuron):
    """
    NAND
    a * -1 + b * -1 >= -1
    """

    def __init__(self, a, b, state=0):
        super(NandNeuron, self).__init__((a, b), (-1, -1), -1, state=state)


class Network(object):
    """
    Helper class for creating and updating functioning networks.

    Inputs must be named.
    `update` returns the states of all terminating nodes as they are given during
    construction.

    Updates in a TOP-DOWN LEFT-RIGHT direction.

    Avoids problems associated with cyclical networks.

    A NOT connected to itself will update in alternating states forever.
    Cool, huh?

    `net["a"]` retrieves an input named "a"
    `net[0]` retrieves the state of the first output
    `list(net)` returns each node in the order it is evaluated (Input/Neuron)
    `net.update(**inputs)` updates inputs by name and returns the result
    """

    def __init__(self, *outputs):
        self.inputs = {}
        self.outputs = outputs
        # collect named inputs for reference by name
        self.size = 0
        for item in self:
            self.size += 1
            if isinstance(item, Input):
                if item.name is None:
                    raise ValueError("Inputs must have names for reference")
                if item.name in self.inputs:
                    raise ValueError("Duplicate input name: %r" % item.name)
                self.inputs[item.name] = item

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.outputs[key]
        else:
            return self.inputs[key]

    def __len__(self):
        return self.size

    def __iter__(self):
        for item in self._iter_helper(self.outputs, []):
            yield item

    def _iter_helper(self, outputs, checked):
        for item in outputs:
            if item not in checked:
                checked.append(item)
                if isinstance(item, Neuron):
                    for input in self._iter_helper(item.inputs, checked):
                        yield input
                yield item

    def update(self, **inputs):
        for name, state in inputs.iteritems():
            self.inputs[name].state = state
        for gate in self:
            if isinstance(gate, Neuron):
                gate.update()
        return tuple(output.state for output in self.outputs)

    def __repr__(self):
        return u"Network(*%r)" % (self.outputs,)


