from Layer import Layer


class MLP:
  def __init__(self, nin: int, nouts):
    """
      @param: nouts 是MLP的每个layer所对应的 output Neurons的个数
    """
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]