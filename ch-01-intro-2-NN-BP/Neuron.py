import random
from Value import Value


class Neuron:
  def __init__(self, nin: int):
    """
      每个Neuron类在初始化的时候会默认创建一个长度为 nin 的 权重向量和 一个bias（单一）值。
      权重的值和bias的值都服从均匀分布，值域在[-1, 1] 之间 (inclusive)
    """
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1, 1))

  def __call__(self, x):
    """
    覆写 n(x) 方法调用。案例：x = [2.0, 3.0]; n = Neuron(x)；__call__覆写的是 n(x)
    该方法就是用来将当前neuron的权重 self.w 与 某个传入节点的 x 值 进行element-wise相乘相加，然后加上自身的 bias self.b
    然后通过一个激活函数激活 并输出。
    算例：
      self.w = [1,2,3]; x = [2,3,5], self.b = -10; 激活函数为tanh
      (1 * 2) + (2 * 3) + (3 * 5) - 10 = 13
    :return => tanh(13)
    """
    # zip 方法是将传入的参数组对。例如 w=[1,2,3]; x=[2,3,5]. list(zip(w, x)) = [(1, 2), (2, 3), (3, 5)]
    act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out
  
  def parameters(self):
    return self.w + [self.b]