import numpy as np
import math
import matplotlib.pyplot as plt


class Value:
  def __init__(self, data, _children=(), _op='', label='') -> None:
    self.data = data
    self.grad = 0.0     # 初始化 梯度为0.0
    self._backward = lambda: None   # a function that doesn't do anything but will be a back-prop function
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self) -> str:
    """
      data wrapper
      用来覆写 print() 方法的输出结果
    """
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    """
      用来覆写 + 的输出结果
    """
    other = other if isinstance(other, Value) else Value(other)   # 处理被加对象不是 Value的情况
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad += 1.0 * out.grad   # 这里使用 += 是为了应对 b = a + a的情况，详见 1:25:40
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out
  
  def __mul__(self, other):
    """
      用来覆写 * 的输出结果
    """
    other = other if isinstance(other, Value) else Value(other)   # 处理被乘对象不是 Value的情况
    out = Value(self.data * other.data, (self, other), '*')
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now... :("
    out = Value(self.data ** other, (self, ), f'** {other}')
    def _backward():
      self.grad += other * (self.data ** (other - 1)) * out.grad   # 幂函数求导
    out._backward = _backward
    return out
  
  def __rmul__(self, other):
    """
      用来覆写乘数不是Value 2 * Value(1.0) 的情况
    """
    return self * other
  
  def __truediv__(self, other):   # self / other
    return self * other ** -1
  
  def __neg__(self):      # define * -1 operation:
    return self * -1
  
  def __sub__(self, other):   # define substraction - operation
    return self + (-other)
    
  def __radd__(self, other):    # define int + Value 的情况
    return self + other
  
  def backward(self):
    topo = []; visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()

  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')
    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward
    return out
  
  def tanh(self):
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = Value(t, (self, ), 'tanh')
    def _backward():
      # o = tanh(n)的导数为： 1 - tanh^2(n)
      self.grad += (1 - t ** 2) * out.grad
    out._backward = _backward
    return out

  def relu(self):
    out = Value(max(0, self.data), (self, ), 'relu')
    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out
  
  
