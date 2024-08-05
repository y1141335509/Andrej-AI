from Neuron import Neuron


class Layer:
  def __init__(self, nin: int, nout: int):
    """
    @param: nout is the number of output Neurons of this Layer
    @param: nin is the number of input Neurons of this Layer
    """
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x):
    """
      覆写的是 Layer(x) 这个函数调用
    """
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]
